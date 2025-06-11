import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.impute import SimpleImputer, KNNImputer
import joblib

# Logging setup
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from ...data.Loader.tabular_dataset import TabularDataset

class TabularBuilderMixin:


    ##############################
    # Building the base dataframe
    ##############################

    # Only with aggregated with per property type per bucket
    def build_aggregated_tabular_feature_df(self, store: bool = False) -> pd.DataFrame:
        """
        Build per-bucket, per-property-type aggregate features.
        For each bucket and property_type, computes:
            - mean of 'mean'
            - mean of 'std'
            - max of 'max'
            - min of 'min'
            # - sum of 'sum'
            - sum of 'count'
            - sum of 'has_measurement'
        Returns a wide-format DataFrame with 'bucket_idx' and columns like '{property_type}_{stat}'.
        """
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("full_feature_df not found; ensure build_full_feature_df() was called.")

        df = self.full_feature_df.copy()

        # Define aggregation functions per feature
        agg_funcs = {
            'mean': 'mean',
            'std': 'mean',
            'max': 'max',
            'min': 'min',
            # 'sum': 'sum',
            'count': 'sum',
            'has_measurement': 'sum'
        }

        # Group by bucket and property
        grouped = (
            df.groupby(['bucket_idx', 'property_type'])
            .agg(agg_funcs)
            .reset_index()
        )

        # Pivot to wide
        pivot = grouped.pivot(index='bucket_idx', columns='property_type')
        # Flatten MultiIndex: (stat, property) -> property_stat
        pivot.columns = [f"{prop}_{stat}" for stat, prop in pivot.columns]
        pivot_df = pivot.reset_index()

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
        Build per-bucket, per-floor, per-property-type aggregate features.
        For each bucket, floor and property_type, computes:
            - mean of 'mean'
            - mean of 'std'
            - max of 'max'
            - min of 'min'
            - sum of 'count'
            - sum of 'has_measurement'
        Returns a wide-format DataFrame with 'bucket_idx' and columns like '{floor}_{property_type}_{stat}'.
        """
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("full_feature_df not found; ensure build_full_feature_df() was called.")

        df = self.full_feature_df.copy()

        # 1) build a mapping from device_uri (string) to floor_number (int)
        device_to_floor = {}
        for dev_uri, dev in self.office_graph.devices.items():
            dev_str = str(dev_uri)
            room_uri = getattr(dev, 'room', None)
            if room_uri is None:
                continue
            # Room.floor is a floor URI; look up its Floor object
            floor_uri = self.office_graph.rooms[room_uri].floor
            floor_obj = self.office_graph.floors.get(floor_uri)
            if floor_obj is None:
                continue
            device_to_floor[dev_str] = floor_obj.floor_number

        # 2) add floor_number column, drop any devices without a floor
        df['floor_number'] = df['device_uri'].map(device_to_floor)
        df = df.dropna(subset=['floor_number'])
        df['floor_number'] = df['floor_number'].astype(int)

        # 3) define the same aggs we use elsewhere
        agg_funcs = {
            'mean': 'mean',
            'std': 'mean',
            'max': 'max',
            'min': 'min',
            'count': 'sum',
            'has_measurement': 'sum'
        }

        # 4) group by bucket, floor, property
        grouped = (
            df
            .groupby(['bucket_idx', 'floor_number', 'property_type'])
            .agg(agg_funcs)
            .reset_index()
        )

        # 5) pivot to wide form
        pivot = grouped.pivot_table(
            index='bucket_idx',
            columns=['floor_number', 'property_type'],
            values=list(agg_funcs.keys())
        )

        # 6) flatten MultiIndex columns: (stat, floor, prop) → "floor_prop_stat"
        pivot.columns = [
            f"{floor}_{prop}_{stat}"
            for stat, floor, prop in pivot.columns
        ]
        result = pivot.reset_index()

        # 7) store or return
        if store:
            self.tabular_feature_df = result
            return None
        else:
            return result

    # With full features, i.e., all measurements as seperate columns
    def build_full_tabular_feature_df(self, store: bool = False) -> pd.DataFrame:
        """
        Pivot the normalized full_feature_df so that each time bucket is a row and
        each (device_uri, property_type, feature) combination becomes its own column.
        Returns:
            A wide-format DataFrame with 'bucket_idx' as one column and device-property features as others.
        """
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("full_feature_df not found; ensure build_full_feature_df() was called.")

        df = self.full_feature_df.copy()
        # Identify feature columns (excluding identifiers)
        id_cols = ['device_uri', 'property_type', 'bucket_idx']
        feature_cols = [c for c in df.columns if c not in id_cols]

        # Pivot to wide format
        wide = df.pivot_table(
            index='bucket_idx',
            columns=['device_uri', 'property_type'],
            values=feature_cols
            # fill_value is omitted, missing entries stay as NaN.
        )
        # Flatten MultiIndex columns
        wide.columns = [f"{dev}_{prop}_{feat}" for dev, prop, feat in wide.columns]
        wide = wide.reset_index()

        # Store the DataFrame if requested
        if store:
            self.tabular_feature_df = wide
            return None
        # Else, return the DataFrame
        else:
            return wide

    def build_combined_tabular_feature_df(self, store: bool = False) -> pd.DataFrame:
        """
        Build both the full and aggregated feature DataFrames and merge them.

        Steps:
          1. Build the full-feature wide DataFrame.
          2. Build the aggregated-per-property wide DataFrame.
          3. Merge on 'bucket_idx' to get one DataFrame with both sets of features.

        Args:
            store (bool): If True, saves the result to self.tabular_feature_df and returns None.
                          If False, returns the merged DataFrame.

        Returns:
            pd.DataFrame or None: The merged DataFrame if store=False, otherwise None.
        """
        # Build the full wide-feature table (no side effect)
        full_df = self.build_full_tabular_feature_df(store=False)

        # Build the aggregated table (no side effect)
        agg_df = self.build_aggregated_tabular_feature_df(store=False)

        # Merge on bucket_idx
        combined = full_df.merge(agg_df, on='bucket_idx', how='left')

        if store:
            self.tabular_feature_df = combined
            return None
        else:
            return combined


    ############################################################
    ################ Integrating existing data #################
    ############################################################


    ##############################
    # Integration: Weather
    ##############################

    def integrate_weather_features(self) -> None:
        """
        Merge weather features into the tabular feature DataFrame on bucket_idx.
        Args:
            weather_dict: Optional dict mapping bucket_idx -> {weather_feature: value}.
                          If None, uses self.weather_data_dict.
        Returns:
            DataFrame with weather columns appended.
        """
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found; call build_tabular_feature_df() first.")
        if not hasattr(self, 'weather_data_dict'):
            raise ValueError("weather_data_dict not found; call get_weather_data() first.")

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
        # There should be no NaNs in weather data, so no need to fillna

        # Store
        self.tabular_feature_df = merged
        return None


    ##############################
    # Integration: Consumption (forecasting target)
    ##############################

    def integrate_consumption_target(self) -> None:
        """
        Merge consumption values as the target column into tabular_with_weather_df.
        Requires self.consumption_values and self.time_buckets to exist.

        Updates the 
        """
        if not hasattr(self, 'time_buckets'):
            raise ValueError("time_buckets not found; ensure initialize_time_parameters() was called.")
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found; call integrate_weather_features() first.")
        if not hasattr(self, 'consumption_values'):
            raise ValueError("consumption_values not found; call get_forecasting_values() first.")

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
        # There are no NaNs in consumption, so no need to fillna

        self.tabular_feature_df = merged
        return None


    ##############################
    # Integration: Time features
    ##############################

    def add_time_features(self) -> None:
        """
        Extract cyclical time features (hour, day of week) from the time buckets.
        Adds 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos' to tabular_feature_df or its variants.
        """
        if not hasattr(self, 'time_buckets'):
            raise ValueError("time_buckets not found; ensure initialize_time_parameters() was called.")

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
    
    
    ############################################################
    ############# Block-aware feature engineering ##############
    ############################################################

    def _assign_block_id(self, df: pd.DataFrame) -> pd.Series:
        """
        Internal helper: map each bucket_idx to its block_id so we can group by block.
        """
        if not hasattr(self, 'blocks'):
            raise ValueError("self.blocks not found, ensure split_time_buckets() was called.")
        
        if not hasattr(self, "_block_map"):
            self._block_map = {
                idx: blk
                for blk, info in self.blocks.items()
                for idx in info["bucket_indices"]
            }
        return df["bucket_idx"].map(self._block_map)

    def add_moving_average_features(self,
                                    windows: List[int] = [3, 6, 12, 24],
                                    shift_amount: int = 0,
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
            cols: which columns to average; defaults to all numeric features.
            shift_amount: how many steps to shift before averaging.
                        - 0: include current bucket (avg over [t-w+1,...,t])
                        - 1: exclude current (avg over [t-w,...,t-1])
                        etc.

        Returns:
            None. Alters the tabular_feature_df in-place with additional columns `<col>_ma_<w>_sh<shift_amount>`.
        """
        # Get the feature DataFrame
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found; call build_tabular_feature_df() first.")
        df = self.tabular_feature_df.copy()

        # Assign block ID to df
        df["block_id"] = self._assign_block_id(df)
        df.sort_values(['block_id', 'bucket_idx'], inplace=True)

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
        gb = df.groupby('block_id')
        for w in windows:
            for col in cols:
                series = (gb[col]
                        .shift(shift_amount)
                        .rolling(window=w, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True))
                moving_average_dict[f"{col}_ma_{w}_sh{shift_amount}"] = series

        # Concatenate all columns in one go
        ma_df = pd.DataFrame(moving_average_dict, index=df.index)
        df = pd.concat([df, ma_df], axis=1)

        # Drop helper
        df.drop(columns='block_id', inplace=True)

        # Defragment
        self.tabular_feature_df = df.copy()
        return None

    def add_lag_features(self,
                        lags: List[int] = [1, 2, 3, 4],
                        cols: Optional[List[str]] = None,
                        use_only_original_columns: bool = False
                        ) -> None:
        """
        For each col in `cols` (default: all numeric except 'bucket_idx'), 
        create lag‐k features per block (no leakage across train/val/test).

        Requirements:
            Must already have self.tabular_feature_df, which have 'bucket_idx' and be sorted by it.

        Args:
            lags: list of integers, e.g. [1, 24]
            cols: which columns to lag; defaults to all numeric features.
        
        Returns:
            None. Alters the tabular_feature_df in-place with additional columns `<col>_lag_<k>`.
        """
        # Get the feature DataFrame
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found; call build_tabular_feature_df() first.")
        df = self.tabular_feature_df.copy()

        # Assign block ID to df
        df["block_id"] = self._assign_block_id(df)
        df.sort_values(['block_id', 'bucket_idx'], inplace=True)
        
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
        gb = df.groupby("block_id")
        for k in lags:
            for col in cols:
                lag_series = gb[col].shift(k).reset_index(level=0, drop=True)
                lag_dict[f"{col}_lag_{k}"] = lag_series

        # Concatenate all columns in one go
        lag_df = pd.DataFrame(lag_dict, index=df.index)
        df = pd.concat([df, lag_df], axis=1)

        # Drop helper
        df.drop(columns='block_id', inplace=True)

        # Defragment
        self.tabular_feature_df = df.copy()
        return None


    ############################################################
    ####################### Imputation #########################
    ############################################################

    def impute_to_tabular(self,
                method: str = 'simple',
                strategy: str = 'mean',
                columns: list = None,
                imputer_kwargs: dict = None) -> None:
        """
        Impute missing values in a DataFrame of features, fitting only on the training split.

        Args:
            df: Optional DataFrame to impute. If None, uses self.tabular_feature_df.
            method: One of {'simple', 'knn'}.
            strategy: For SimpleImputer, one of {'mean', 'median', 'most_frequent', 'constant'}.
            columns: List of column names to impute. Defaults to all numeric except 'bucket_idx'.
            imputer_kwargs: Additional keyword args passed to the chosen imputer.

        Returns:
            The imputed DataFrame.
        """

        # 1) Obtain DataFrame
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError('No tabular_feature_df available for imputation.')
        df = self.tabular_feature_df.copy()

        # 2) Determine columns to impute
        if columns is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude 'bucket_idx' and 'consumption' (target variable) if present
            if 'bucket_idx' in num_cols:
                num_cols.remove('bucket_idx')
            if 'consumption' in num_cols:
                num_cols.remove('consumption')
            columns = num_cols

        # 3) Instantiate chosen imputer
        imputer_kwargs = imputer_kwargs or {}
        if method == 'simple':
            imputer = SimpleImputer(strategy=strategy, **imputer_kwargs)
        elif method == 'knn':
            imputer = KNNImputer(**imputer_kwargs)
        else:
            raise ValueError(f"Unknown imputation method '{method}'")

        # 4) Fit imputer on training split
        if not hasattr(self, 'train_indices'):
            raise ValueError('train_indices not set; call split_time_buckets first.')
        train_idx = self.train_indices
        X_train = df.iloc[train_idx][columns]
        imputer.fit(X_train)

        # 5) Transform all splits
        df[columns] = imputer.transform(df[columns])

        # 6) Write back in place
        self.tabular_feature_df = df
        return None

    ############################################################
    ################ Preparing Tabular Input ###################
    ############################################################

    def create_and_save_tabular_dataset(self, save_path = "data/processed/tabular_dataset.joblib"):
        """
        Convenience wrapper: build the final DataFrame (with features + target),
        then wrap into TabularDataset with the existing train/val/test indices.
        
        Args:
            task. "forecasting" (consumption values), or "classification" (workhour labels)
        """
        df = self.tabular_feature_df.copy()

        # Create the consumption df
        consumption_df = df[["bucket_idx", "consumption"]].copy()
        consumption_df["block_id"] = self._assign_block_id(consumption_df)
        consumption_df.sort_values(['block_id', 'bucket_idx'], inplace=True)

        dataset = TabularDataset(
            features_df=df,
            consumption_df = consumption_df,
            workhour_labels = self.workhour_labels,
            train_idx=self.train_indices,
            val_idx=self.val_indices,
            test_idx=self.test_indices,
            blocks=self.blocks
        )

        dataset.save(path=save_path)
        return None