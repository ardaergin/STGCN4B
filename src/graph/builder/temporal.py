import sys
import numpy as np
from typing import Dict
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging
import matplotlib.pyplot as plt
from rdflib import URIRef

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from ..officegraph import OfficeGraph

class TemporalBuilderMixin:
    office_graph: OfficeGraph

    def initialize_time_parameters(self, 
                                  start_time: str = "2022-03-07 00:00:00", # Monday
                                  # The data starts at 03-01 (Tuesday), but we start at 03-07 (Monday 00:00)
                                  end_time: str = "2023-01-29 00:00:00", # 01-29, Sunday
                                  # The data ends at 01-31 (Tuesday), but we end at 01-29 (Monday 00:00)
                                  interval: str   = "1h",
                                  use_sundays: bool = False) -> None:
        """
        Initialize time-related parameters and create time buckets.
        
        Args:
            start_time: Start time for analysis in format "YYYY-MM-DD HH:MM:SS"
            end_time: End time for analysis in format "YYYY-MM-DD HH:MM:SS"
            interval: Frequency (15min, 30min, 1h, 2h…)
            use_sundays: Whether to include Sundays in time buckets
        """
        self.start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        self.interval = interval
        self.use_sundays = use_sundays

        # Build buckets at arbitrary freq (15min, 30min, 1h, 2h, …)
        full_index = pd.date_range(self.start_time,
                                   self.end_time,
                                   freq=self.interval,
                                   inclusive="left")
        if not self.use_sundays:
            full_index = full_index[full_index.weekday != 6] # Sunday = 6
        
        # store as list of (start, end)
        off = pd.tseries.frequencies.to_offset(self.interval)
        self.time_buckets = [
            (ts.to_pydatetime(), (ts + off).to_pydatetime())
            for ts in full_index
        ]
        
        logger.info(f"Created {len(self.time_buckets)} buckets at frequency {self.interval}")
        return None
    
    def split_time_buckets(self,
                        train_blocks=3,
                        val_blocks=1,
                        test_blocks=1,
                        seed=2658918) -> None:
        """
        Split time buckets into train, validation, and test sets using block-wise sampling.
        Takes the direct number of blocks for each split and requires that the total requested
        blocks can divide evenly into the available blocks.
        
        Args:
            train_blocks: Number of blocks for training
            val_blocks: Number of blocks for validation
            test_blocks: Number of blocks for testing
            seed: Random seed for reproducibility
            
        Raises:
            ValueError: If the number of available blocks is not divisible by the 
                        total requested blocks
        """
        # -------------------------------------------------------------------------
        # 1) Initialization check and seed
        # -------------------------------------------------------------------------
        if not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")
        if seed is not None:
            np.random.seed(seed)
        
        # -------------------------------------------------------------------------
        # 2) Computing how many buckets per block
        # -------------------------------------------------------------------------
        time_indices = list(range(len(self.time_buckets)))

        # First computing how many buckets per day, then per week (or 6-day)
        # (so 96/day for 15min, 48/day for 30T, 24/day for 1H, etc.)
        offset = pd.Timedelta(self.interval)
        buckets_per_day = int(pd.Timedelta("1D") / offset)

        # Then per week (6-day if sundays excluded, 7-day if included)
        days_per_block = 7 if self.use_sundays else 6
        block_size = buckets_per_day * days_per_block

        logger.info(f"Using {days_per_block}-day blocks, {block_size} buckets at {self.interval} each.")

        # -------------------------------------------------------------------------
        # 3) Building the list of blocks (each block is a list of bucket‐indices)
        # -------------------------------------------------------------------------
        initial_blocks = []
        for i in range(0, len(time_indices), block_size):
            block = time_indices[i : i + block_size] # Taking up to block_size indices (last block might be smaller)
            initial_blocks.append(block)

        n_blocks = len(initial_blocks)
        block_types = [None] * n_blocks  # will hold "train"/"val"/"test" for each block_id

        logger.info(f"Created {n_blocks} blocks of data (each {days_per_block} days)")
                
        # -------------------------------------------------------------------------
        # 4) Deciding how many blocks to assign to train/val/test
        # -------------------------------------------------------------------------
        total_requested_blocks = train_blocks + val_blocks + test_blocks

        # If n_blocks is not divisible by total_requested, mark some extra as train
        n_extra_blocks = n_blocks % total_requested_blocks
        if n_extra_blocks != 0:
            logger.warning(
                f"The number of blocks ({n_blocks}) is not divisible by "
                f"train+val+test=({total_requested_blocks}). "
                f"So, {n_extra_blocks} extra block(s) will be randomly chosen,"
                f"and will be forcibly assigned to train."
            )
            # Randomly choosing which blocks become “extra train”
            extra_block_ids = np.random.choice(np.arange(n_blocks), size=n_extra_blocks, replace=False)
            for block in extra_block_ids:
                block_types[block] = "train"
            # the remaining blocks (n_blocks - n_extra) will be split evenly among train/val/test
            remaining_block_ids = [block for block in range(n_blocks) if block not in set(extra_block_ids)]
        else:
            # no extras: everything gets assigned exactly in the next step
            remaining_block_ids = list(range(n_blocks))

        # -------------------------------------------------------------------------
        # 5) Build the repeated pattern for the remaining blocks
        # -------------------------------------------------------------------------

        # How many blocks remain to assign in (train, val, test) pattern?
        n_remain = len(remaining_block_ids)
        # Each “chunk” of (train_blocks, val_blocks, test_blocks) is total_requested
        repeat_factor = n_remain // total_requested_blocks

        basic_pattern = ["train"] * train_blocks + ["val"] * val_blocks + ["test"] * test_blocks
        sampling_pattern = basic_pattern * repeat_factor
        np.random.shuffle(sampling_pattern)

        # Assign each of the remaining blocks to a split, in randomized order
        for idx_in_list, block_id in enumerate(remaining_block_ids):
            block_types[block_id] = sampling_pattern[idx_in_list]

        # -------------------------------------------------------------------------
        # 6) Now collect bucket‐indices per split, and build self.blocks dict
        # -------------------------------------------------------------------------
        train_indices = []
        val_indices = []
        test_indices = []

        # Build the final self.blocks mapping
        self.blocks = {}
        for block_id, bucket_list in enumerate(initial_blocks):
            btype = block_types[block_id]  # "train"/"val"/"test"
            # Guard against any block_type being None (shouldn't happen unless mis‐count)
            if btype is None:
                raise RuntimeError(f"Block {block_id} was never assigned a type.")

            # Add its bucket indices to the right global‐indices list
            if btype == "train":
                train_indices.extend(bucket_list)
            elif btype == "val":
                val_indices.extend(bucket_list)
            else:  # "test"
                test_indices.extend(bucket_list)

            # Populate self.blocks[block_id]
            self.blocks[block_id] = {
                "block_type": btype,
                "bucket_indices": list(bucket_list),  # copy to avoid aliasing
            }

        # -------------------------------------------------------------------------
        # 7) Sort each global list so that train/val/test indices are in ascending order
        # -------------------------------------------------------------------------
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()

        # Save them on the instance
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        # Logging summary
        # Count how many blocks ended up in each split:
        n_train_blocks = sum(1 for t in block_types if t == "train")
        n_val_blocks   = sum(1 for t in block_types if t == "val")
        n_test_blocks  = sum(1 for t in block_types if t == "test")
        logger.info(
            f"Data split: "
            f"Train={len(train_indices)} buckets ({n_train_blocks} blocks), "
            f"Val={len(val_indices)} buckets ({n_val_blocks} blocks), "
            f"Test={len(test_indices)} buckets ({n_test_blocks} blocks)"
        )
        logger.info(f"Pattern of train/val/test per block: {block_types}")
        return None

    #############################
    # Normalization Helper
    #############################

    def _fit_and_apply_scaler(self,
                              X: np.ndarray,
                              name: str,
                              scaler_type: str,
                              train_index: np.ndarray = None) -> np.ndarray:
        """
        Fit (if needed) and apply a scikit-learn scaler to X.
        By default, fits on rows given by self.train_indices, but a custom
        train_index array can be provided (boolean mask or integer indices).
        Caches the scaler under attribute '{name}_scaler_'.

        Args:
            X: Array of shape (n_samples, n_features).
            name: Identifier for this scaler (e.g. 'weather', 'consumption', 'measurements_{prop}').
            scaler_type: One of {'standard', 'robust', 'minmax'}.
            train_index: Optional array of row indices or boolean mask to select training rows.

        Returns:
            X_transformed: Array of same shape as X.
        """
        logger = logging.getLogger(__name__)

        valid = {'standard', 'robust', 'minmax'}
        if scaler_type not in valid:
            raise ValueError(f"Unknown scaler '{scaler_type}', choose from {valid}")

        # Determine training rows
        if train_index is None:
            if not hasattr(self, 'train_indices'):
                raise ValueError('train_indices not set; call split_time_buckets first')
            train_index = self.train_indices
        # boolean mask
        if hasattr(train_index, 'dtype') and train_index.dtype == bool:
            X_train = X[train_index]
        else:
            X_train = X[train_index]
        if X_train.size == 0:
            raise ValueError(f"No training data available to fit scaler for '{name}'")

        scaler_attr = f"{name}_scaler_"
        if not hasattr(self, scaler_attr):
            # Initialize scaler
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = MinMaxScaler()
            scaler.fit(X_train)
            setattr(self, scaler_attr, scaler)
            logger.info(f"Fitted {scaler_type} scaler for {name} on {X_train.shape[0]} samples")
        else:
            scaler = getattr(self, scaler_attr)

        # Transform all data
        return scaler.transform(X)

    #############################
    # Weather Data (Predictor)
    #############################

    def get_weather_data(self,
                        weather_csv_path: str = "data/weather/hourly_weather_2022_2023.csv",
                        normalize: bool = True,
                        scaler: str = "robust"
                        ) -> Dict[int, Dict[str, float]]:
        """
        Load, aggregate, feature-engineer, and optionally normalize
        weather data for forecasting.

        Args:
            weather_csv_path: CSV with hourly weather (must have a 'date' column).
            normalize: If True, return scaled features; otherwise raw.
            scaler: One of {"standard", "robust", "minmax"}.

        Returns:
            Dict[bucket_idx → {feature_name: value}].
        """
        if not hasattr(self, "train_indices"):
            raise ValueError("Call split_time_buckets first.")

        from ...data.Weather.weather import (
            load_weather_csv,
            get_weather_data_for_time_buckets
        )

        # 1) Load & aggregate per‐bucket
        df = load_weather_csv(weather_csv_path, self.start_time, self.end_time)
        bucket_weather = get_weather_data_for_time_buckets(
            df, self.time_buckets, self.interval
        )
        if not bucket_weather:
            logger.warning("No weather data for any bucket.")
            return {}

        # 2) Build DataFrame (rows=buckets, cols=raw features)
        weather_df = pd.DataFrame.from_dict(bucket_weather, orient="index")

        # 3) Feature engineering
        # — wind directions → sin & cos
        for col in ("wind_direction_10m", "wind_direction_80m"):
            if col in weather_df:
                θ = np.deg2rad(weather_df[col].astype(float))
                weather_df[f"{col}_sin"] = np.sin(θ)
                weather_df[f"{col}_cos"] = np.cos(θ)
                weather_df.drop(columns=[col], inplace=True)

        # — one-hot encode weather_code
        if "weather_code" in weather_df:
            weather_df = pd.get_dummies(weather_df,
                                        columns=["weather_code"],
                                        prefix="wc",
                                        dtype=float)

        # 4) Early return of raw features
        if not normalize:
            # keep for later if you want
            self.weather_features_ = weather_df
            return weather_df.to_dict(orient="index")

        # 5) Normalizing the numeric features
        valid = {"standard", "robust", "minmax"}
        if scaler not in valid:
            raise ValueError(f"Unknown scaler `{scaler}`; choose from {valid}")

        numeric_cols = [
            'temperature_2m',
            'relative_humidity_2m',
            'precipitation',
            'wind_speed_10m',
            'wind_speed_80m',
            'cloud_cover'
        ]
        angle_cols = [col for col in weather_df.columns if col.endswith('_sin') or col.endswith('_cos')]
        cat_cols   = [c for c in weather_df.columns if c.startswith('wc_')]

        # 1) Scale only numeric columns
        X_num = weather_df[numeric_cols].values
        X_num_scaled = self._fit_and_apply_scaler(X_num, 'weather_numeric', scaler)

        # 2) Reassemble a new DataFrame
        scaled_df = pd.DataFrame(
            data=X_num_scaled,
            index=weather_df.index,
            columns=numeric_cols
        )
        # bring in the unscaled angle and categorical columns
        for col in angle_cols + cat_cols:
            scaled_df[col] = weather_df[col]

        # 3) Build your final dict
        normed = {
            idx: scaled_df.iloc[idx].to_dict()
            for idx in range(len(scaled_df))
        }
        self.weather_features_norm_ = normed
        return normed
    
    #############################
    # Task-Specific (Target) Data
    #############################
    
    def get_classification_labels(self, country_code: str = 'NL') -> np.ndarray:
        """
        Generate work hour classification labels for each time bucket.
        
        Args:
            country_code: Country code for holidays
            
        Returns:
            Binary labels for each time bucket (1 for work hour, 0 for non-work hour)
        """
        # Check if time buckets are available
        if not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")
        
        # Initialize work hour classifier
        from ...data.TimeSeries.workhours import WorkHourClassifier
        work_hour_classifier = WorkHourClassifier(
            country_code=country_code,
            start_year=self.start_time.year,
            end_year=self.end_time.year
        )
        
        # Classify time buckets
        labels_list = work_hour_classifier.classify_time_buckets(self.time_buckets)
        
        # Convert to numpy array
        labels = np.array(labels_list, dtype=int)
        
        logger.info(f"Generated {len(labels)} classification labels")
        logger.info(f"Work hours: {labels.sum()} ({labels.sum()/len(labels):.1%} of time buckets)")
        
        return labels
    
    def get_forecasting_values(self,
                            consumption_dir: str = "data/consumption",
                            normalize: bool = True,
                            scaler: str = "robust") -> Dict[int, float]:
        """
        Load, aggregate (to time buckets), and optionally normalize
        consumption data for forecasting using scikit-learn scalers.

        Args:
            consumption_dir: Directory containing consumption data files
            normalize: if True, return normalized consumption values,
                       else return raw consumption.
            scaler: Type of scaler to use ("standard", "robust", "minmax")

        Returns:
            Dictionary mapping time bucket index to consumption value
            (normalized if normalize=True).
        """
        if not hasattr(self, "train_indices"):
            raise ValueError("Call split_time_buckets first.")
            
        valid_scalers = {"standard", "robust", "minmax"}
        if scaler not in valid_scalers:
            raise ValueError(f"Unknown scaler `{scaler}`, choose from {valid_scalers}")

        # 1) load and aggregate exactly as before
        from ...data.TimeSeries.consumption import (
            load_consumption_files,
            aggregate_consumption_to_time_buckets
        )
        consumption_data = load_consumption_files(
            consumption_dir,
            self.start_time,
            self.end_time
        )
        bucket_consumption = aggregate_consumption_to_time_buckets(
            consumption_data,
            self.time_buckets,
            self.interval
        )
        # now bucket_consumption: { idx: raw_value }

        # 2) If asked for raw, just return the dict
        if not normalize:
            logger.info(f"Returning raw consumption for {len(bucket_consumption)} buckets")
            return bucket_consumption
        # Else, we normalize the data...

        # 3) Convert to array format for scikit-learn
        T = len(self.time_buckets)
        consumption_array = np.array([bucket_consumption[i] for i in range(T)], dtype=float).reshape(-1, 1)

        # 4) Normalize
        X_norm = self._fit_and_apply_scaler(consumption_array, 'consumption', scaler)
        normed = {i: float(X_norm[i, 0]) for i in range(T)}
        return normed

    #############################
    # Measurement bucketing
    #############################

    def bucket_measurements_by_device_property(self, drop_sum: bool = True) -> None:
        """
        Bucket measurements by device and property according to time buckets,
        using a single pass and pandas groupby for speed.

        Returns:
            DataFrame with columns:
                ['device_uri', 'property_type', 'bucket_idx',
                'mean', 'std', 'max', 'min', 'sum', 'count']
        """
        if not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")

        # 1) Flatten all measurements into a DataFrame
        records = []
        ignored = getattr(self, "ignored_property_types", set())
        for device_uri, device in self.office_graph.devices.items():
            for prop_uri in device.properties:
                # determine property_type
                property_type = next(
                    (pt for pt, uris in self.office_graph.property_type_mappings.items()
                    if prop_uri in uris),
                    None
                )
                if not property_type or property_type in ignored:
                    continue
                for meas in device.measurements_by_property.get(prop_uri, []):
                    ts = meas.timestamp
                    if not (self.start_time <= ts < self.end_time):
                        continue
                    records.append({
                        "device_uri": str(device_uri),
                        "property_type": property_type,
                        "timestamp": ts,
                        "value": meas.value
                    })

        df = pd.DataFrame.from_records(records)
        if df.empty:
            # No data: return empty DataFrame with correct columns
            cols = ["device_uri","property_type","bucket_idx",
                    "mean","std","max","min","sum","count"]
            empty_df = pd.DataFrame(columns=cols)
            self.bucketed_measurements_df = empty_df
            return empty_df

        # 2) Assign each measurement to a time bucket using pd.cut
        bin_edges = [start for start, _ in self.time_buckets] + [self.time_buckets[-1][1]]
        labels = list(range(len(self.time_buckets)))
        df["bucket_idx"] = pd.cut(df["timestamp"],
                                bins=bin_edges,
                                right=False,
                                labels=labels)
        df = df.dropna(subset=["bucket_idx"])
        df["bucket_idx"] = df["bucket_idx"].astype(int)

        # 3) Aggregate with pandas groupby
        agg_df = (
            df.groupby(["device_uri","property_type","bucket_idx"])["value"]
            .agg(["mean","std","max","min","sum","count"])
            .reset_index()
        )

        # 4) Null‐out stats for 'Contact'
        contact_mask = agg_df["property_type"] == "Contact"
        agg_df.loc[contact_mask, ["mean","std","max","min"]] = np.nan

        # 5) Pick columns (drop 'sum' if requested)
        cols = ["device_uri","property_type","bucket_idx",
                "mean","std","max","min","sum","count"]
        if drop_sum:
            cols.remove("sum")
        result_df = agg_df[cols]

        self.bucketed_measurements_df = result_df

        # --- Sanity logging ---
        logger.info("-" * 40)
        logger.info(f"BUCKETING SUMMARY ({len(result_df)} rows total)\n")

        # Unique vs expected
        n_buckets = result_df['bucket_idx'].nunique()
        expected = len(self.time_buckets)
        logger.info(f"• Buckets covered: {n_buckets}, {expected}\n")

        # Missing buckets
        all_bins = set(range(expected))
        present = set(result_df['bucket_idx'].unique())
        missing = sorted(all_bins - present)
        logger.info(f"• Number of missing buckets: {len(missing)}\n")

        # distribution of records per bucket
        counts = result_df.groupby('bucket_idx').size()
        desc = counts.describe()
        logger.info(
            f"Per-bucket record counts summary: count={desc['count']}, mean={desc['mean']:.2f}, "
            f"std={desc['std']:.2f}, min={desc['min']}, 25%={desc['25%']}, "
            f"50%={desc['50%']}, 75%={desc['75%']}, max={desc['max']}"
        )

        # Per‐device stats
        logger.info("ROWS PER DEVICE:")
        for dev, cnt in result_df['device_uri'].value_counts().items():
            ub = result_df[result_df['device_uri']==dev]['bucket_idx'].nunique()
            logger.info(f"  - {dev}: {cnt} rows in {ub} buckets")
        logger.info("")

        # Per‐property stats
        logger.info("ROWS PER PROPERTY:")
        for prop, cnt in result_df['property_type'].value_counts().items():
            ub = result_df[result_df['property_type']==prop]['bucket_idx'].nunique()
            logger.info(f"  - {prop}: {cnt} rows in {ub} buckets")

        # sample row
        sample = result_df.iloc[0].to_dict()
        logger.info(f"Sample row: {sample}")

        logger.info("-" * 40)
        return None
    
    def visualize_bucket_distributions(self):
        """
        Visualize histograms of each feature for every (device_uri, property_type),
        and display a table of how many buckets are filled per device-property.
        """
        from IPython.display import display

        # Features to plot
        features = ["mean", "std", "max", "min", "sum", "count"]
        
        # 1) Count how many buckets are filled per device-property
        counts = (
            self.bucketed_measurements_df
            .groupby(["device_uri", "property_type"])["bucket_idx"]
            .nunique()
            .reset_index(name="filled_buckets")
        )
        # Display counts table
        print("Filled buckets per device-property:")
        display(counts)
        
        # 2) Plot histogram for each device-property-feature
        for (device_uri, prop), grp in self.bucketed_measurements_df.groupby(["device_uri", "property_type"]):
            for feature in features:
                if feature not in grp.columns:
                    continue
                values = grp[feature].dropna()
                if values.empty:
                    continue
                plt.figure(figsize=(6,4))
                plt.hist(values, bins=30)
                plt.title(f"{device_uri} | {prop} | {feature}")
                plt.xlabel(feature)
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.show()
        
    #############################
    # Normalization of Bucketed Measurements
    #############################

    def normalize_bucketed_measurements(self, drop_sum: bool = True, scaler: str = "robust") -> None:
        """
        Normalize bucketed measurements per property type.

        Args:
            drop_sum: If True, drop the 'sum' feature before scaling.
            scaler: One of {'standard','robust','minmax'} to choose scaling method.

        Returns:
            DataFrame of normalized measurements, same shape as bucketed_measurements_df.
        """
        if not hasattr(self, 'bucketed_measurements_df'):
            raise ValueError("Call bucket_measurements_by_device_property first.")
        if not hasattr(self, 'train_indices'):
            raise ValueError("Call split_time_buckets first.")

        df = self.bucketed_measurements_df.copy()
        norm_dfs = []

        # Track overall train usage
        total_train_used = 0

        for prop, group in df.groupby('property_type'):
            # Decide features per property
            if drop_sum:
                feats = ['mean', 'std', 'max', 'min', 'count']
            else:
                feats = ['sum', 'count'] if prop == 'Contact' else ['mean', 'std', 'max', 'min', 'sum', 'count']

            # Build feature matrix X and training mask
            X = group[feats].values
            train_mask = group['bucket_idx'].isin(self.train_indices).values

            n_train = int(train_mask.sum())
            total_train_used += n_train
            logger.info(f"Property '{prop}': using {n_train}/{len(group)} rows for training on {group['bucket_idx'].nunique()} unique buckets")

            # Normalize using helper
            X_scaled = self._fit_and_apply_scaler(
                X,
                f'measurements_{prop}',
                scaler,
                train_index=train_mask
            )

            # Rebuild group with scaled features
            norm_group = group[['device_uri', 'property_type', 'bucket_idx']].reset_index(drop=True)
            norm_group[feats] = X_scaled
            norm_dfs.append(norm_group)

        # Concatenate all properties
        normalized_df = pd.concat(norm_dfs, ignore_index=True)
        self.normalized_measurements_df = normalized_df

        # Logging summary
        total_rows = len(normalized_df)
        unique_buckets = normalized_df['bucket_idx'].nunique()
        logger.info(f"Normalized measurements: {total_rows} rows across {unique_buckets} unique buckets")
        logger.info(f"Total training buckets used across all properties: {total_train_used}")
        return None
        
    #############################
    # Adding back the empty buckets
    #############################

    def build_full_feature_df(self) -> None:
        """
        Re-index the normalized measurements so you get one row per
        (device_uri, property_type, bucket_idx), filling missing stats with 0,
        and adding a binary 'has_measurement' indicator.
        
        Args:
            drop_sum: If True, omit the 'sum' column from the final output.

        Returns:
            DataFrame with columns:
            ['device_uri', 'property_type', 'bucket_idx', <feature_cols>, 'has_measurement']
        """
        if not hasattr(self, "normalized_measurements_df"):
            raise ValueError("Call normalize_bucketed_measurements first.")

        # 1) Tag existing normalized rows as present
        df = self.normalized_measurements_df.copy()
        df['has_measurement'] = 1.0

        # 2) Build full grid index
        full_idx = pd.MultiIndex.from_product(
            [
                df['device_uri'].unique(),
                df['property_type'].unique(),
                range(len(self.time_buckets))
            ],
            names=["device_uri", "property_type", "bucket_idx"]
        )

        # 3) Reindex and reset
        full_df = (
            df
            .set_index(['device_uri', 'property_type', 'bucket_idx'])
            .reindex(full_idx)
            .reset_index()
        )

        # 4) Choose features and zero-fill stats for the missing bucket-device-property combinations
        feature_cols = ['mean', 'std', 'max', 'min', 'count']
        full_df[feature_cols] = full_df[feature_cols].fillna(0.0)
        ### (!) Took out the part above (!)
        ### Because I will do his later in the HeteroGraphBuilderMixin and not in TabularBuilderMixin

        # 5) Fill NaNs with 0.0 for the 'has_measurement', 
        # since the rows we just added are not measured anyways
        full_df['has_measurement'] = full_df['has_measurement'].fillna(0.0)

        # --- Sanity logging ---
        logger.info("=" * 40)
        logger.info(f"FULL FEATURE MATRIX ({len(full_df)} rows)")

        # Devices, properties, buckets
        n_devices = full_df['device_uri'].nunique()
        n_props = full_df['property_type'].nunique()
        n_buckets = full_df['bucket_idx'].nunique()
        logger.info(f"• Devices: {n_devices}")
        logger.info(f"• Property types: {n_props}")
        logger.info(f"• Time buckets: {n_buckets}")

        # Sparsity: proportion of rows without measurements
        missing_mask = full_df['has_measurement'] == 0.0
        missing_pct = missing_mask.mean() * 100
        logger.info(f"• Missing entries: {missing_mask.sum()} rows ({missing_pct:.1f}% of total)")

        # Sample row
        sample = full_df.iloc[0].to_dict()
        logger.info(f"Sample row (first): {sample}")

        logger.info("=" * 40)

        self.full_feature_df = full_df
        return None