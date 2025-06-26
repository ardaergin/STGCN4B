import os 
import pandas as pd
import numpy as np
from typing import Optional, List
import joblib

# Logging setup
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TabularBuilderMixin:

    ############################################################
    # Building the base dataframe for building-wide prediction tasks:
    # - workhour_classification
    # - consumption_forecast
    # 
    # For these tasks, every row should represent a time bucket

    # On the other hand, for building the base dataframe for measurement_forecast task,
    # every row should represent a time_bucket-room_uri pair. This structure is already there 
    # in self.room_feature_df
    ############################################################

    # Only with aggregated with per property type per bucket
    def build_aggregated_tabular_feature_df(self, store: bool = False) -> pd.DataFrame:
        """
        Builds per-bucket, building-wide aggregate features for each property type.

        For each bucket and property_type, computes:
            - mean of 'mean'
            - mean of 'std'
            - max of 'max'
            - min of 'min'
            - sum of 'count'
            - sum of 'has_measurement'
            - max of 'has_measurement'

        The resulting DataFrame has one row per `bucket_idx` and wide-format
        columns like `{property_type}_{stat}` (e.g., 'Temperature_mean',
        'Temperature_n_devices').

        The final DataFrame is stored in `self.tabular_feature_df` or returned.
        """
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("full_feature_df not found; ensure build_full_feature_df() was called.")
        df = self.full_feature_df.copy()

        # Use pivot_table directly for a cleaner implementation
        pivot = pd.pivot_table(
            df,
            index='bucket_idx',
            columns='property_type',
            values=['mean', 'std', 'max', 'min', 'count', 'has_measurement'],
            aggfunc={
                'mean': 'mean',
                'std': 'mean',
                'max': 'max',
                'min': 'min',
                'count': 'sum',
                'has_measurement': ['sum', 'max']
            }
        )

        # Flatten the multi-level columns
        new_cols = []
        for col in pivot.columns:
            # col is a tuple, e.g., ('mean', 'Temperature') or ('has_measurement', 'sum', 'Temperature')
            if col[0] == 'has_measurement':  # Special case with 3 levels
                agg_type, prop_type = col[1], col[2]
                stat_name = 'n_devices' if agg_type == 'sum' else 'has_measurement'
                new_cols.append(f"{prop_type}_{stat_name}")
            else:  # Standard case with 2 levels
                stat, prop_type = col[0], col[1]
                new_cols.append(f"{prop_type}_{stat}")
        pivot.columns = new_cols
        pivot_df = pivot.reset_index()

        logger.info(f"Created aggregated tabular feature DataFrame. Shape: {pivot_df.shape}")

        # Store the DataFrame if requested
        if store:
            self.tabular_feature_df = pivot_df
            return None
        # Else, return the DataFrame
        else:
            return pivot_df
    
    # With floor aggregation
    def build_floor_aggregated_tabular_feature_df(self, store: bool = False) -> pd.DataFrame:
        """
        Builds per-bucket, per-floor, per-property-type aggregate features.
        
        For each bucket and property_type, computes:
            - mean of 'mean'
            - mean of 'std'
            - max of 'max'
            - min of 'min'
            - sum of 'count'
            - sum of 'has_measurement'
            - max of 'has_measurement'
        
        The final DataFrame is stored in `self.tabular_feature_df` or returned.
        """
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("full_feature_df not found; ensure build_full_feature_df() was called.")
        df = self.full_feature_df.copy()

        # 1) build a mapping from device_uri (string) to floor_number (int)
        device_to_floor = {}
        for dev_uri, dev in self.office_graph.devices.items():
            dev_str = str(dev_uri)
            room_uri = getattr(dev, 'room', None)
            room_obj = self.office_graph.rooms.get(room_uri)
            if room_obj is None:
                continue
            # Room.floor is a floor URI; look up its Floor object
            floor_uri = room_obj.floor
            floor_obj = self.office_graph.floors.get(floor_uri)
            if floor_obj is None:
                continue
            device_to_floor[dev_str] = floor_obj.floor_number

        # 2) add floor_number column, drop any devices without a floor
        df['floor_number'] = df['device_uri'].map(device_to_floor)
        df = df.dropna(subset=['floor_number'])
        df['floor_number'] = df['floor_number'].astype(int)

        # 3) Use a single pivot_table call to aggregate and reshape
        pivot = pd.pivot_table(
            df,
            index='bucket_idx',
            columns=['floor_number', 'property_type'],
            values=['mean', 'std', 'max', 'min', 'count', 'has_measurement'],
            aggfunc={
                'mean': 'mean',
                'std': 'mean',
                'max': 'max',
                'min': 'min',
                'count': 'sum',
                'has_measurement': ['sum', 'max']
            }
        )

        # 4) Flatten the multi-level columns
        # col is now a tuple like ('mean', 0, 'Temperature') or ('has_measurement', 'sum', 0, 'Temperature')
        new_cols = []
        for col in pivot.columns:
            if col[0] == 'has_measurement':
                agg_type, floor, prop = col[1], col[2], col[3]
                stat_name = 'n_devices' if agg_type == 'sum' else 'has_measurement'
                new_cols.append(f"F{floor}_{prop}_{stat_name}")
            else:
                stat, floor, prop = col[0], col[1], col[2]
                new_cols.append(f"F{floor}_{prop}_{stat}")

        pivot.columns = new_cols
        pivot_df = pivot.reset_index()

        # 5) Store or return
        if store:
            self.tabular_feature_df = pivot_df
            logger.info(f"Stored floor-aggregated tabular feature DataFrame. Shape: {pivot_df.shape}")
            return None
        else:
            return pivot_df

    # With full features, i.e., all measurements as seperate columns
    def build_full_tabular_feature_df(self, store: bool = False) -> pd.DataFrame:
        """
        Pivot the full_feature_df to a wide format where each time bucket is a row.

        Each (device_uri, property_type, feature) combination becomes its own column.
        This creates a very wide, sparse DataFrame suitable for models that can
        handle high dimensionality.
        
        The final DataFrame is stored in `self.tabular_feature_df` or returned.
        """
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("full_feature_df not found; ensure build_full_feature_df() was called.")

        df = self.full_feature_df.copy()
        # Identify feature columns (excluding identifiers)
        id_cols = ['device_uri', 'property_type', 'bucket_idx']
        feature_cols = [c for c in df.columns if c not in id_cols]

        # Pivot to wide format
        pivot = df.pivot_table(
            index='bucket_idx',
            columns=['device_uri', 'property_type'],
            values=feature_cols
        )
        # Flatten MultiIndex columns
        pivot.columns = [f"{dev}_{prop}_{feat}" for feat, dev, prop in pivot.columns]
        pivot_df = pivot.reset_index()

        # Store the DataFrame if requested
        if store:
            self.tabular_feature_df = pivot_df
            return None
        # Else, return the DataFrame
        else:
            return pivot_df
    
    ############################################################
    ################ Integrating existing data #################
    ############################################################

    ##############################
    # Integration: Weather
    ##############################

    def integrate_weather_features(self) -> None:
        """
        Merge weather features into self.tabular_feature_df on bucket_idx.
        Requires self.tabular_feature_df, self.weather_data_dict, and self.time_buckets to exist.
        """
        if not hasattr(self, 'time_buckets'):
            raise ValueError("time_buckets not found; ensure initialize_time_parameters() was called.")
        if not hasattr(self, 'weather_data_dict'):
            raise ValueError("weather_data_dict not found; call get_weather_data() first.")
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found. Get it first.")

        # Convert weather dict to DataFrame
        weather_df = pd.DataFrame.from_dict(self.weather_data_dict, orient='index')
        weather_df = weather_df.reset_index().rename(columns={'index': 'bucket_idx'})

        # Merge on bucket_idx
        merged = pd.merge(
            self.tabular_feature_df,
            weather_df,
            on='bucket_idx',
            how='left'
        )

        # Store
        self.tabular_feature_df = merged
        return None


    ##############################
    # Integration: Consumption (forecasting target)
    ##############################

    def integrate_consumption_target(self) -> None:
        """
        Merge consumption values as the target column into self.tabular_feature_df on bucket_idx.
        Requires self.tabular_feature_df, self.consumption_values, and self.time_buckets to exist.
        """
        if not hasattr(self, 'time_buckets'):
            raise ValueError("time_buckets not found; ensure initialize_time_parameters() was called.")
        if not hasattr(self, 'consumption_values'):
            raise ValueError("consumption_values not found; call get_forecasting_values() first.")
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found. Get it first.")

        # Build consumption DataFrame
        n = len(self.time_buckets)
        cons_df = pd.DataFrame({
            'bucket_idx': list(range(n)),
            'consumption': self.consumption_values
        })
        merged = pd.merge(
            self.tabular_feature_df,
            cons_df,
            on='bucket_idx',
            how='left'
        )

        self.tabular_feature_df = merged
        return None
    
    ##############################
    # Integration: Time features
    ##############################

    def add_time_features(self) -> None:
        """
        Extract cyclical time features (hour, day of week) from the time buckets.
        Adds 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos' to tabular_feature_df.
        """
        if not hasattr(self, 'time_buckets'):
            raise ValueError("time_buckets not found; ensure initialize_time_parameters() was called.")
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found. Get it first.")

        # Map bucket_idx to start timestamp
        ts_map = {i: tb[0] for i, tb in enumerate(self.time_buckets)}
        df = getattr(self, 'tabular_feature_df')
        df['timestamp'] = df['bucket_idx'].map(ts_map)
        dt = pd.to_datetime(df['timestamp'])

        # Hour of day
        hours = dt.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        # Day of week
        dows = dt.dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dows / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dows / 7)
        
        # Drop helper timestamp
        df.drop(columns=['timestamp'], inplace=True)
        return None
    
    ##############################
    # Integration: Room features
    ##############################
    
    def add_static_room_features(self) -> None:
        """
        Adds static room attributes (e.g., area, hasWindows) to the tabular feature DataFrame.

        This method is only applicable for tasks where the DataFrame has a 'room_uri' column,
        such as 'measurement_forecast'. It creates a lookup table of static features for each
        room and merges them into the main feature set.
        """
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("`tabular_feature_df` not found. Build it first.")
        
        logger.info(f"Adding {len(self.static_room_attributes)} static room features...")

        # 1. Create a list of dictionaries, one for each room in your graph.
        room_data = []
        for room_uri, room_obj in self.office_graph.rooms.items():
            features = {'room_uri': room_uri}
            for attr_string in self.static_room_attributes:
                if attr_string == "floor":
                    floor_uri = room_obj.floor
                    floor_obj = self.office_graph.floors.get(floor_uri)
                    floor_num = getattr(floor_obj, "floor_number", np.nan)
                    features["floor"] = floor_num
                else:
                    val = self._get_nested_attr(room_obj, attr_string, default=np.nan)
                    features[attr_string] = val
            room_data.append(features)

        if not room_data:
            logger.warning("Could not gather any static room data.")
            return None

        # 2. Convert the list of dicts into a clean DataFrame.
        static_features_df = pd.DataFrame(room_data)

        # 3. Merge this static data into the main feature DataFrame.
        self.tabular_feature_df = pd.merge(
            self.tabular_feature_df,
            static_features_df,
            on='room_uri',
            how='left'  # Use a left merge to keep all rows from the original df.
        )

        logger.info("Successfully merged static room features.")
        return None
    
    ############################################################
    ############# Block-aware feature engineering ##############
    ############################################################

    def _assign_block_id(self, df: pd.DataFrame) -> pd.Series:
        """
        Internal helper: map each bucket_idx to its block_id so we can group by block.
        """
        if not hasattr(self, 'blocks'):
            raise ValueError("self.blocks not found, ensure build_weekly_blocks() was called.")
        
        if not hasattr(self, "_block_map"):
            self._block_map = {
                idx: blk
                for blk, info in self.blocks.items()
                for idx in info["bucket_indices"]
            }
        return df["bucket_idx"].map(self._block_map)

    def add_moving_average_features(self,
                                    windows: List[int],
                                    shift_amount: int = 0,
                                    extra_grouping_cols: Optional[List[str]] = None,
                                    cols: Optional[List[str]] = None,
                                    use_only_original_columns: bool = False
                                    ) -> None:
        """
        For each col in `cols` (default: all numeric except 'bucket_idx'),
        create backward-looking moving averages of the past values,
        per block (so no cross‐block leakage).
        
        Requirements:
            Must already have self.tabular_feature_df, which have 'bucket_idx' and be sorted by it.
        
        Args:
            windows: list of window sizes, e.g. [3, 24].
            shift_amount: how many steps to shift before averaging.
                        - 0: include current bucket (avg over [t-w+1,...,t])
                        - 1: exclude current (avg over [t-w,...,t-1])
                        etc.
            extra_grouping_cols: any additional grouping columns (like 'room_uri'), additional to ['block_id'].
            cols: which columns to average; defaults to all numeric features.
            use_only_original_columns: whether to only consider columns that were present in the original `tabular_feature_df` 
                                        (i.e., not columns generated by previous calls to `add_moving_average_features` or `add_lag_features`).
                                        This prevents creating MAs of MAs or MAs of lags.

        Returns:
            None. Alters the tabular_feature_df in-place with additional columns `<col>_ma_<w>_sh<shift_amount>`.
        """
        # Get the feature DataFrame
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found. Get it first.")
        df = self.tabular_feature_df.copy()

        # Assign block ID to df
        df["block_id"] = self._assign_block_id(df)

        # Grouping
        grouping = ['block_id']
        if extra_grouping_cols:
            grouping.extend(extra_grouping_cols)
        logger.info(f"Adding moving average features, grouping by: {grouping}")

        # Sorting based on grouping + bucket_idx
        df.sort_values(grouping + ['bucket_idx'], inplace=True)

        # Pick which columns to do moving average
        if cols is None:
            # start from every numeric column except bucket_idx/block_id
            all_num = df.select_dtypes("number").columns.tolist()
            candidates = [c for c in all_num if c not in ("bucket_idx", "block_id")]
            if use_only_original_columns:
                # filter out any generated MA or lag columns
                cols = [
                    c for c in candidates
                    if "_ma_" not in c and "_lag_" not in c
                ]
            else:
                cols = candidates

        # compute everything into a dict of Series
        moving_average_dict = {}
        gb = df.groupby(grouping)
        for w in windows:
            for col in cols:
                series = (gb[col]
                        .shift(shift_amount)
                        .rolling(window=w, min_periods=1)
                        .mean())
                moving_average_dict[f"{col}_ma_{w}_sh{shift_amount}"] = series

        # Concatenate all columns in one go
        ma_df = pd.DataFrame(moving_average_dict, index=df.index)
        df = pd.concat([df, ma_df], axis=1)

        # Drop helper
        df.drop(columns='block_id', inplace=True)

        # Defragment
        self.tabular_feature_df = df
        return None

    def add_lag_features(self,
                        lags: List[int],
                        extra_grouping_cols: Optional[List[str]] = None,
                        cols: Optional[List[str]] = None,
                        use_only_original_columns: bool = False,
                        ) -> None:
        """
        For each col in `cols` (default: all numeric except 'bucket_idx'), 
        create lag‐k features per block (no leakage across train/val/test).

        Requirements:
            Must already have self.tabular_feature_df, which have 'bucket_idx' and be sorted by it.

        Args:
            lags: list of integers, e.g. [1, 24]
            extra_grouping_cols: any additional grouping columns (like 'room_uri'), additional to ['block_id'].
            cols: which columns to lag; defaults to all numeric features.
            use_only_original_columns: whether to only consider columns that were present in the original `tabular_feature_df` 
                                        (i.e., not columns generated by previous calls to `add_moving_average_features` or `add_lag_features`).
                                        This prevents creating MAs of MAs or MAs of lags.
        
        Returns:
            None. Alters the tabular_feature_df in-place with additional columns `<col>_lag_<k>`.
        """
        # Get the feature DataFrame
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found. Get it first.")
        df = self.tabular_feature_df.copy()

        # Assign block ID to df
        df["block_id"] = self._assign_block_id(df)

        # Grouping
        grouping = ['block_id']
        if extra_grouping_cols:
            grouping.extend(extra_grouping_cols)
        logger.info(f"Adding lag features, grouping by: {grouping}")

        # Sorting based on grouping + bucket_idx
        df.sort_values(grouping + ['bucket_idx'], inplace=True)

        # Pick which columns to lag
        if cols is None:
            # start from every numeric column except bucket_idx/block_id
            all_num = df.select_dtypes("number").columns.tolist()
            candidates = [c for c in all_num if c not in ("bucket_idx", "block_id")]
            if use_only_original_columns:
                # filter out any generated MA or lag columns
                cols = [
                    c for c in candidates
                    if "_ma_" not in c and "_lag_" not in c
                ]
            else:
                cols = candidates

        # For each lag and each column
        lag_dict = {}
        gb = df.groupby(grouping)
        for k in lags:
            for col in cols:
                lag_series = gb[col].shift(k)
                lag_dict[f"{col}_lag_{k}"] = lag_series

        # Concatenate all columns in one go
        lag_df = pd.DataFrame(lag_dict, index=df.index)
        df = pd.concat([df, lag_df], axis=1)

        # Drop helper
        df.drop(columns='block_id', inplace=True)

        # Defragment
        self.tabular_feature_df = df
        return None
    
    ############################################################
    ################### Preparing Targets ######################
    ############################################################

    def create_target_consumption(self, horizon: int = 1) -> None:
        """
        Creates the future consumption target by shifting the consumption values
        backwards within each block to prevent data leakage.

        This method adds a new 'target_consumption' column to `self.tabular_feature_df`.
        Rows where the target could not be created (i.e., the last `horizon` rows
        of each block) will have NaN and must be dropped before training.

        Args:
            horizon (int): The number of time steps into the future to predict.
                        Defaults to 1 (predict the next time step).
        """
        if not hasattr(self, 'tabular_feature_df') or 'consumption' not in self.tabular_feature_df.columns:
            raise ValueError("Run `integrate_consumption_target` first to add the 'consumption' column.")
        
        logger.info(f"Creating block-aware consumption target for horizon={horizon}...")

        df = self.tabular_feature_df.copy()

        # 1. Assign block ID for grouping
        df['block_id'] = self._assign_block_id(df)

        # 2. Group by block and shift to get the future value
        #    .shift(-horizon) pulls future values into the current row.
        df['target_consumption'] = df.groupby('block_id')['consumption'].shift(-horizon)

        # 3. Drop the helper column
        df.drop(columns='block_id', inplace=True)
        
        # 4. Report on the number of NaNs created
        nan_count = df['target_consumption'].isna().sum()
        logger.info(f"Created 'target_consumption'. Found {nan_count} rows with NaN targets (these should be dropped).")
        
        self.tabular_df = df
        return None
    
    def create_target_measurement(self, stat: str = "mean", horizon: int = 1) -> None:
        """
        Creates the future measurement target for a specific variable (e.g., Temperature).
        
        This is done by shifting the measurement values backwards within each block AND for
        each room to prevent data leakage. It adds a 'target_measurement' column.

        Args:
            stat (str): The statistic of the measurement variable to use as the target (e.g., 'mean').
            horizon (int): The number of time steps into the future to predict. Defaults to 1.
        """
        if not hasattr(self, 'tabular_feature_df') or 'room_uri' not in self.tabular_feature_df.columns:
            raise ValueError("This method is for measurement forecasting and requires 'room_uri' in the DataFrame.")
        
        # The source column from which we create the target
        source_column = f"{self.measurement_variable}_{stat}"
        if source_column not in self.tabular_feature_df.columns:
            raise ValueError(f"Source column '{source_column}' not found in the DataFrame.")
            
        logger.info(f"Creating block-aware measurement target from '{source_column}' for horizon={horizon}...")

        df = self.tabular_feature_df.copy()

        # 1. Assign block ID for grouping
        df['block_id'] = self._assign_block_id(df)

        # 2. IMPORTANT: Group by both block and room, then shift
        df['target_measurement'] = df.groupby(['block_id', 'room_uri'])[source_column].shift(-horizon)

        # 3. Drop the helper column
        df.drop(columns='block_id', inplace=True)

        # 4. Report on the number of NaNs
        nan_count = df['target_measurement'].isna().sum()
        logger.info(f"Created 'target_measurement'. Found {nan_count} total rows with NaN targets.")

        self.tabular_df = df
        return None

    ############################################################
    #################### Master Function #######################
    ############################################################

    def build_tabular_df(self, 
                         forecast_horizon: int = 1,
                         lags: List[int] = None,
                         windows: List[int] = None,
                         shift_amount: int = 1):
        """
        This is the final, all-in-one function to build tabular data depending on the 'build_mode' (or 'task_type').
        1) Handles building the base DataFrame, appropriate to the task type
        2) Does feature engineering.
        3) Adds the appropriate target column to the DataFrame, and cleans the rows where target is NaN.
        """

        ##### Building the base DataFrame #####

        if self.build_mode == "workhour_classification":
            # We have only a single floor (floor 7) for this task
            # So we can just build the full, then also get the aggregates
            full_df = self.build_full_tabular_feature_df(store=False)
            agg_df = self.build_aggregated_tabular_feature_df(store=False)
            combined_df = full_df.merge(agg_df, on='bucket_idx', how='left')
            self.tabular_feature_df = combined_df
        
        elif self.build_mode == "consumption_forecast":
            # This is building-level consumption. 
            # Using all devices (160), and adding MA and lag, etc. would be too much features
            # Per-floor level aggregation is the best for this scenario
            # And like before, just add the aggregates
            per_floor_df = self.build_floor_aggregated_tabular_feature_df(store=False)
            agg_df = self.build_aggregated_tabular_feature_df(store=False)
            combined_df = per_floor_df.merge(agg_df, on='bucket_idx', how='left')
            self.tabular_feature_df = combined_df

            # For this specific task, we can add consumption already
            # We want to also create time and lag feature for this
            self.integrate_consumption_target()

        else: # self.build_mode == "measurement_forecast":
            self.tabular_feature_df = self.room_feature_df.copy()


        ##### Feature engineering #####

        # Adding weather features here so we get also their MAs and lags 
        self.integrate_weather_features()

        # Defining selective feature lists for lags and moving averages
        base_cols = self.tabular_feature_df.select_dtypes("number").columns.tolist()
        base_cols = [c for c in base_cols if c not in ('bucket_idx', 'block_id')]

        # Tier 1: Core signals for both Lags and MAs
        # All means, maxes, mins, and key weather variables
        core_signals_for_lags_and_ma = [
            c for c in base_cols if 
            any(k in c for k in ['_mean', '_max', '_min']) or
            c in ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m', 'wind_speed_80m', 'cloud_cover']
        ]
        logger.info(f"Generating lags and MAs for {len(core_signals_for_lags_and_ma)} core signal columns.")

        # Tier 2: Secondary signals for MA only
        secondary_signals_for_ma_only = [
            c for c in base_cols if 
            any(k in c for k in ['_std', '_count', '_n_devices', '_has_measurement'])
        ]
        logger.info(f"Generating MAs only for {len(secondary_signals_for_ma_only)} secondary signal columns.")

        if self.build_mode == "measurement_forecast":
            # For this task, we also have "room_uri" as an additional grouping col
            extra_grouping_col = ["room_uri"]
        else:
            extra_grouping_col = None

        # Defaults for MAs & Lags
        if lags is None:
            lags=[1, 2, 3]
        if windows is None:
            windows=[3, 6, 12, 24]

        # Creating MAs & Lags
        self.add_lag_features(
            extra_grouping_cols=extra_grouping_col,
            lags=lags,
            cols=core_signals_for_lags_and_ma)
        
        self.add_moving_average_features(
            extra_grouping_cols=extra_grouping_col,
            windows=windows,
            shift_amount=shift_amount,
            cols=core_signals_for_lags_and_ma + secondary_signals_for_ma_only)

        # NOTE 1: We can add the time features after taking MA & lag,
        #         as we should not really take the lag of the time-related features

        # NOTE 2: We should exclude time-related columns for the workhour_classification
        #         as time features are direct leakage for the classifying classifying
        #         whether an hour is work hour or not.
        if self.build_mode != "workhour_classification":
            self.add_time_features()
        
        # Feature engineering: add static room features for measurement_forecast task
        if self.build_mode == "measurement_forecast":
            self.add_static_room_features()

            # Dealing with categorical features
            for col in ['room_uri', 'isProperRoom',
                        'hasWindows', 'has_multiple_windows']:
                self.tabular_feature_df[col] = self.tabular_feature_df[col].astype(str).astype('category')

            # Coding floor both as a category and as a numeric
            self.tabular_feature_df["floor_num"] = self.tabular_feature_df["floor"].astype(int)
            self.tabular_feature_df["floor_cat"] = self.tabular_feature_df["floor"].astype("category")
            self.tabular_feature_df = self.tabular_feature_df.drop("floor", axis=1)

            # NOTE: For measurement_forecast, the has_measurement columns are binary
            #       So, turning them to categorical make sense.
            #       This is not the case for other build_mode / task types, 
            #       as we summed over them, and they are continuous in those cases.
            #       So, I am only doing it here, in this if branch.
            has_measurement_cols = [c for c in self.tabular_feature_df.columns
                                    if "has_measurement" in c]
            self.tabular_feature_df[has_measurement_cols] = (
                self.tabular_feature_df[has_measurement_cols]
                    .astype("category")
            )

        ##### Target preparation #####
        # 1. Create the appropriate target column based on the task
        if self.build_mode == "workhour_classification":
            if not hasattr(self, 'workhour_labels'):
                raise ValueError("workhour_labels attribute not found for classification task.")
            self.target_col_name = 'workhour_labels'
            self.tabular_df = self.tabular_feature_df.copy()
            self.tabular_df[self.target_col_name] = self.workhour_labels
        
        elif self.build_mode == "consumption_forecast":
            self.target_col_name = 'target_consumption'
            self.create_target_consumption(horizon=forecast_horizon)
            # The function above creates the new self.tabular_df

        elif self.build_mode == "measurement_forecast":
            self.target_col_name = 'target_measurement'
            self.create_target_measurement(horizon=forecast_horizon)
            # The function above creates the new self.tabular_df
        
        else:
            raise ValueError(f"Unknown build_mode: {self.build_mode}")
        
        df = self.tabular_df

        # 2. CRITICAL: Drop all rows where the target is NaN.
        # For both consumption_forecast & measurement_forecast, we have:
        # - end-of-block NaNs
        # For measurement_forecast, we have:
        # - missing measurements in certain time buckets, so NaNs in target cols.
        initial_rows = len(df)
        df.dropna(subset=[self.target_col_name], inplace=True)
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN targets. {len(df)} rows remaining.")

        # Defragmented copy
        self.tabular_df = df.copy() 

        if df.empty:
            raise ValueError("No valid data rows remaining after dropping NaN targets.")
        
        return None

    def save_tabular_df(self, output_path: str) -> None:
        """
        Saves tabular_df.
        """
        if not hasattr(self, 'tabular_df'):
            raise ValueError("tabular_df not found. Run build_tabular_df() first.")
        
        logger.info(f"Saving tabular data for task: {self.build_mode}")

        # 3. Create the final payload to save
        tabular_input = {
            "df": self.tabular_df,
            "target_col_name": self.target_col_name,
            "blocks": self.blocks,
            "task_type": self.build_mode
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the dictionary using joblib
        joblib.dump(tabular_input, output_path)
        logger.info(f"Successfully saved final DataFrame to {output_path}")
        return None