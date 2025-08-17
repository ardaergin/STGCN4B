from typing import Dict, Optional, List
import os 
import numpy as np
from datetime import datetime
import pandas as pd

import logging; logger = logging.getLogger(__name__)


class TemporalBuilderMixin:

    def initialize_time_parameters(
            self, 
            start_time: str = "2022-03-07 00:00:00",
            end_time: str = "2023-01-29 00:00:00",
            interval: str   = "30min",
            use_sundays: bool = False
    ) -> None:
        """
        Initialize time-related parameters and create time buckets.
        
        Args:
            start_time: Start time for analysis in format "YYYY-MM-DD HH:MM:SS"
            end_time: End time for analysis in format "YYYY-MM-DD HH:MM:SS"
            interval: Frequency (15min, 30min, 1h, 2h…)
            use_sundays: Whether to include Sundays in time buckets

        Notes:
            The data starts at 03-01 (Tuesday), but we start at 03-07 (Sunday/Monday 00:00)
            The data ends at 01-31 (Tuesday), but we end at 01-29 (Sunday/Monday 00:00)
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
    
    def build_weekly_blocks(self) -> None:
        """
        Groups time buckets into sequential, non-overlapping blocks (e.g., weeks).

        This method inspects the `time_buckets` and groups them into larger
        blocks of a fixed size. For example, all buckets corresponding to a 
        6-day or 7-day period are grouped into a single block.
        
        This method creates the `self.blocks` attribute, a dictionary where keys
        are block IDs and values contain the list of bucket indices for that block.
        """
        if not hasattr(self, 'time_buckets') or not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")

        # 1) Computing how many buckets per block
        time_indices = list(range(len(self.time_buckets)))

        # First computing how many buckets per day, then per week (or 6-day)
        # (so 96/day for 15min, 48/day for 30T, 24/day for 1H, etc.)
        offset = pd.Timedelta(self.interval)
        buckets_per_day = int(pd.Timedelta("1D") / offset)

        # Then per week (6-day if sundays excluded, 7-day if included)
        days_per_block = 7 if self.use_sundays else 6
        block_size = buckets_per_day * days_per_block
        self.block_size = block_size

        logger.info(f"Using {days_per_block}-day blocks, {block_size} buckets at {self.interval} each.")

        # 2) Building the list of blocks (each block is a list of bucket‐indices)
        initial_blocks = []
        for i in range(0, len(time_indices), block_size):
            block = time_indices[i : i + block_size] # Taking up to block_size indices (last block might be smaller)
            initial_blocks.append(block)

        n_blocks = len(initial_blocks)

        logger.info(f"Created {n_blocks} blocks of data (each {days_per_block} days)")
                
        # 3) Store the blocks in a dictionary
        self.blocks = {}
        for block_id, bucket_list in enumerate(initial_blocks):
            self.blocks[block_id] = {
                "bucket_indices": list(bucket_list)
            }
        
        return None
    
    #############################
    # Weather Data (Predictor)
    #############################

    def get_weather_data(
            self,
            weather_csv_path: str = "data/weather/hourly_weather_2022_2023.csv",
            add_weather_code_onehot_features: bool = False
    ) -> None:
        """
        Load, aggregate, and feature-engineer weather data for forecasting.

        Args:
            weather_csv_path: CSV with hourly weather (must have a 'date' column).

        Returns:
            None, but stores the weather data in `self.weather_df` and `self.weather_feature_names`.
        """        
        logger.info("Loading and processing weather data...")
        
        # 1) Load & aggregate per‐bucket
        from ...data.Weather.weather import load_weather_csv, get_weather_data_for_time_buckets
        df = load_weather_csv(weather_csv_path, self.start_time, self.end_time)
        weather_dict = get_weather_data_for_time_buckets(
            weather_df=df, time_buckets=self.time_buckets, interval=self.interval
        )
        if not weather_dict:
            raise ValueError("Weather CSV produced no data. Check path or date range.")
        
        # 2) Build DataFrame (rows=buckets, cols=raw features)
        weather_df = pd.DataFrame.from_dict(weather_dict, orient="index")
        weather_df = weather_df.reset_index().rename(columns={'index': 'bucket_idx'})
        
        # 3) Feature engineering
        # Wind directions → sin & cos
        for col in ("wind_direction_10m", "wind_direction_80m"):
            if col in weather_df:
                θ = np.deg2rad(weather_df[col].astype(float))
                weather_df[f"{col}_sin"] = np.sin(θ)
                weather_df[f"{col}_cos"] = np.cos(θ)
                weather_df.drop(columns=[col], inplace=True)

        # One-hot encode weather_code
        if add_weather_code_onehot_features:
            if "weather_code" in weather_df:
                weather_df = pd.get_dummies(weather_df,
                                            columns=["weather_code"],
                                            prefix="wc",
                                            dtype=float)
            else:
                raise ValueError("`weather_code` column not found in weather_df, "
                                 "but add_weather_code_onehot_features is True.")
        else:
            if "weather_code" in weather_df:
                weather_df.drop(columns=["weather_code"], inplace=True)
        
        self.weather_feature_names = [
            c for c in weather_df.columns
            if c != 'bucket_idx'
        ]
        self.weather_df = weather_df
        return None
    
    def get_workhour_labels(
            self, 
            country_code: str = 'NL', 
            workhour_start: int = 8,
            workhour_end: int = 18,
            save: bool = True,
    ) -> None:
        """
        Generate work hour labels for each time bucket.
        
        Args:
            country_code: Country code for holidays.
            workhour_start: start hour for workhours.
            workhour_end: end hour for workhours.
        
        Returns:
            None, but stores the labels in `self.workhour_labels_df`.
        """
        if not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")
        logger.info("Generating work hour labels...")
        
        # Initialize work hour classifier & classify time buckets
        from ...data.TimeSeries.workhours import WorkHourClassifier
        classifier = WorkHourClassifier(
            country_code=country_code,
            start_year=self.start_time.year,
            end_year=self.end_time.year,
            workhour_start=workhour_start,
            workhour_end=workhour_end
        )
        labels_dict: Dict[int, int] = classifier.classify_time_buckets(
            self.time_buckets
        )
        
        # Build the DataFrame
        # NOTE: We know every bucket_idx is present.
        workhour_df = (
            pd.DataFrame.from_dict(labels_dict, orient="index", columns=["is_workhour"])
            .reset_index()
            .rename(columns={"index": "bucket_idx"})
            .astype({"is_workhour": "int8"})
        )
        self.workhour_labels_df = workhour_df
        
        # Logging
        n_labels = len(workhour_df)
        work_hours   = workhour_df["is_workhour"].sum()
        work_percent = work_hours / n_labels
        logger.info(f"Generated {n_labels} work‑hour labels "
                    f"({work_hours} work, {work_percent:.1%} of buckets).")
        
        # (Optional) Export
        if save:
            file_name = f'workhours_{self.interval}.parquet'
            file_full_path = os.path.join(self.processed_data_dir, file_name)
            logger.info(f"Saving workhour DataFrame to: {file_full_path}")
            self.workhour_labels_df.to_parquet(file_full_path, index=False)

        return None
    
    #############################
    # Task-Specific (Target) Data
    #############################
        
    def get_consumption_values(
            self, 
            consumption_dir: str = "data/consumption", 
            save: bool = True,
    ) -> None:
        """
        Load and aggregate consumption data for forecasting.

        Args:
            consumption_dir: Directory containing consumption data files
            save: whether to save the np.array to disk.

        Returns:
            None, but stores the consumption values in `self.consumption_df`.
        """        
        logger.info("Loading and processing consumption data...")
        from ...data.TimeSeries.consumption import (
            load_consumption_files,
            aggregate_consumption_to_time_buckets
        )
        # Load and aggregate
        consumption_data = load_consumption_files(
            consumption_dir,
            self.start_time,
            self.end_time
        )
        consumption_dict: Dict[int, float] = aggregate_consumption_to_time_buckets(
            consumption_data,
            self.time_buckets,
            self.interval
        )
        
        # Build the DataFrame
        T = len(self.time_buckets)
        consumption_df = (
            pd.DataFrame.from_dict(consumption_dict, orient="index", columns=["consumption"])
            .reindex(range(T)) # Guarantees every bucket_idx row
            .reset_index()
            .astype({"consumption": "float32"})
            .rename(columns={"index": "bucket_idx"})
        )
        self.consumption_df   = consumption_df
        
        # Logging
        n_vals     = len(consumption_df)
        n_missing  = consumption_df["consumption"].isna().sum()
        mean_cons  = consumption_df["consumption"].mean()
        logger.info(
            f"Loaded {n_vals} consumption values "
            f"({n_missing} missing, mean = {mean_cons:.2f})."
        )

        # (Optional) Export
        if save:
            file_name = f'target_consumption_{self.interval}.parquet'
            file_full_path = os.path.join(self.processed_data_dir, file_name)
            logger.info(f"Saving consumption DataFrame to: {file_full_path}")
            self.consumption_df.to_parquet(file_full_path, index=False)
        
        return None
    
    ##############################
    # Property-level DataFrame
    ##############################

    def build_property_level_df(self) -> None:
        """
        Build a DataFrame containing raw property measurements without time bucketing.
        
        This method creates a foundational DataFrame with columns:
        - timestamp: The exact timestamp of the measurement
        - device_uri: The device URI as a string
        - property_type: The property type (e.g., Temperature, CO2Level, Humidity)
        - value: The measurement value
        
        The resulting DataFrame contains only valid measurements within the specified
        time range and for the selected property types. No aggregation or bucketing
        is performed at this stage.
        
        Stores the result in `self.property_level_df`.
        """
        logger.info("Building property-level DataFrame with raw measurements...")
        
        # Create property URI to type mapping for efficient lookups
        prop_uri_to_type = {
            uri: p_type
            for p_type, uris in self.office_graph.property_type_mappings.items()
            for uri in uris
        }
        
        # Collect all valid measurements
        records = []
        total_measurements = 0
        skipped_out_of_range = 0
        skipped_wrong_property = 0
        
        for meas in self.office_graph.measurements.values():
            total_measurements += 1
            
            # Filter by time range
            if not (self.start_time <= meas.timestamp < self.end_time):
                skipped_out_of_range += 1
                continue
            
            # Filter by property type
            prop_type = prop_uri_to_type.get(meas.property_type)
            if not prop_type or prop_type not in self.used_property_types:
                skipped_wrong_property += 1
                continue
            
            # Add valid measurement
            records.append({
                "timestamp": meas.timestamp,
                "device_uri": str(meas.device_uri),
                "property_type": prop_type,
                "value": meas.value
            })
        
        if not records:
            raise ValueError(
                f"No measurements found within the specified time range "
                f"({self.start_time} to {self.end_time}) and for the given properties "
                f"({self.used_property_types}). Total measurements: {total_measurements}, "
                f"Out of range: {skipped_out_of_range}, Wrong property: {skipped_wrong_property}"
            )
        
        # Create DataFrame
        df = pd.DataFrame.from_records(records)
        
        # Drop any rows with NaN values
        initial_rows = len(df)
        df.dropna(subset=['value'], inplace=True)
        rows_with_nan = initial_rows - len(df)
        
        # Convert categorical columns for memory efficiency
        df['device_uri'] = df['device_uri'].astype('category')
        df['property_type'] = df['property_type'].astype('category')
        
        # Store the DataFrame
        self.property_level_df = df
        
        # Calculate and log statistics
        n_measurements = len(df)
        n_unique_devices = df['device_uri'].nunique()
        n_unique_properties = df['property_type'].nunique()
        
        logger.info("=" * 60)
        logger.info("PROPERTY-LEVEL DATAFRAME STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total measurements processed: {total_measurements:,}")
        logger.info(f"  - Skipped (out of time range): {skipped_out_of_range:,}")
        logger.info(f"  - Skipped (wrong property type): {skipped_wrong_property:,}")
        logger.info(f"  - Dropped (NaN values): {rows_with_nan:,}")
        logger.info(f"Valid measurements retained: {n_measurements:,}")
        logger.info(f"Unique devices: {n_unique_devices}")
        logger.info(f"Property types: {n_unique_properties} ({', '.join(df['property_type'].unique())})")
        
        # Log average frequency of measurments per property type
        # Step 1: Define a function to calculate avg interval for a single device-property group
        def calculate_avg_interval(group):
            if len(group) < 2:
                return pd.NaT  # Cannot calculate an interval with less than 2 points
            # Sort by time, calculate difference between consecutive timestamps, and get the mean
            return group['timestamp'].sort_values().diff().mean()

        # Step 2: Apply this function to each device-property group
        avg_intervals_per_device = df.groupby(['device_uri', 'property_type']).apply(calculate_avg_interval)

        # Step 3: Now, average those results across devices for each property type
        avg_intervals_by_prop = avg_intervals_per_device.groupby(level='property_type').mean()
        
        logger.info("-" * 60)
        logger.info("Average measurements per property type:")
        for prop_type in sorted(df['property_type'].unique()):
            prop_data = df[df['property_type'] == prop_type]
            avg_value = prop_data['value'].mean()
            std_value = prop_data['value'].std()
            count = len(prop_data)
            pct = (count / n_measurements) * 100
            
            # Use the pre-calculated, accurate average interval
            avg_interval_str = ""
            avg_delta = avg_intervals_by_prop.get(prop_type)
            if pd.notna(avg_delta):
                avg_seconds = avg_delta.total_seconds()
                avg_interval_str = f", avg interval = {avg_seconds:,.1f}s"

            logger.info(
                f"  {prop_type:15s}: {count:7,} measurements ({pct:5.1f}%), "
                f"avg value = {avg_value:8.2f} (±{std_value:.2f}){avg_interval_str}"
            )
                
        return None
    
    
    ##############################
    # Device-level DataFrame
    ##############################
    
    def build_device_level_df(self) -> None:
        """
        Constructs a complete device-level feature DataFrame.
        
        This function uses the property_level_df as its foundation and performs:
        1. Time bucketing of raw measurements
        2. Aggregation into statistical features
        3. Grid expansion to cover all device-property-bucket combinations
        4. Imputation of known values (count=0 for missing, std=0 for single measurements)
        
        The final DataFrame is stored in `self.device_level_df`.
        """
        if not hasattr(self, 'time_buckets'):
            raise ValueError("time_buckets not found. Run initialize_time_parameters() first.")
        if not hasattr(self, 'property_level_df'):
            raise ValueError("property_level_df not found. Run build_property_level_df() first.")
        df = self.property_level_df.copy()
        
        logger.info("Starting to build the device-level feature DataFrame...")
        
        # === Stage 1: Bucketing ===
        logger.info("[Step 1/4] Assigning measurements to time buckets...")
        bin_edges = [start for start, _ in self.time_buckets] + [self.time_buckets[-1][1]]
        labels = list(range(len(self.time_buckets)))
        df["bucket_idx"] = pd.cut(df["timestamp"], bins=bin_edges, right=False, labels=labels)
        
        # Drop measurements that don't fall into any bucket (shouldn't happen, but safety check)
        df.dropna(subset=["bucket_idx"], inplace=True)
        df["bucket_idx"] = df["bucket_idx"].astype(int)
        
        # === Stage 2: Aggregation ===
        logger.info("[Step 2/4] Aggregating measurements by device, property, and time bucket...")
        stats = ["mean", "std", "max", "min", "count"]
        agg_df = df.groupby(["device_uri", "property_type", "bucket_idx"])["value"].agg(stats).reset_index()
        
        # === Stage 3: Grid Expansion ===
        logger.info("[Step 3/4] Expanding to a full grid and reindexing...")
        full_idx = pd.MultiIndex.from_product(
            [
                agg_df['device_uri'].unique(),
                agg_df['property_type'].unique(),
                range(len(self.time_buckets))
            ],
            names=["device_uri", "property_type", "bucket_idx"]
        )
        full_df = agg_df.set_index(['device_uri', 'property_type', 'bucket_idx']).reindex(full_idx).reset_index()
        
        # Rename device_uri to device_uri_str for consistency with the rest of the codebase
        full_df.rename(columns={'device_uri': 'device_uri_str'}, inplace=True)
        
        # === Stage 4: Imputation & Feature Creation ===
        logger.info("[Step 4/4] Performing imputation and creating features...")
        
        # For newly created rows, 'count' is NaN and can be safely filled with 0
        full_df['count'] = full_df['count'].fillna(0.0)
        
        # Create binary flag for 'has_measurement'
        full_df['has_measurement'] = (full_df['count'] > 0).astype(float)
        
        # Fix std for single measurements (when count=1, std should be 0)
        count_is_one = full_df['count'] == 1
        full_df.loc[count_is_one, 'std'] = 0.0

        # === Logging ===
        logger.info("=" * 40)
        logger.info(f"BUILT FULL DEVICE FEATURE MATRIX ({len(full_df)} rows)")
        
        # Has measurement mask
        missing_mask = full_df['has_measurement'] == 0.0
        missing_pct = missing_mask.mean() * 100
        logger.info(f"Sparsity: {missing_mask.sum()} entries ({missing_pct:.1f}%) have no valid measurements.")
        
        # Count
        total_rows = len(full_df)
        count_zero = (full_df['count'] == 0).sum()
        count_one = (full_df['count'] == 1).sum()
        count_many = (full_df['count'] > 1).sum()
        
        logger.info(
            f"No measurements (count=0):    {count_zero:10,} "
            f"({(count_zero/total_rows):.1%})"
        )
        logger.info(
            f"Single measurement (count=1): {count_one:10,} "
            f"({(count_one/total_rows):.1%})"
        )
        logger.info(
            f"Multiple measurements (count>1): {count_many:9,} "
            f"({(count_many/total_rows):.1%})"
        )
        
        # Variation
        count_std_zero = (full_df['std'] == 0).sum()
        
        # Check where mean, min, and max are all equal (ignores NaNs automatically)
        equal_stats_mask = (full_df['mean'] == full_df['min']) & (full_df['mean'] == full_df['max'])
        count_equal_stats = equal_stats_mask.sum()

        logger.info("-" * 20 + " Variation Statistics " + "-" * 20)
        logger.info(
            f"Instances with std == 0:        {count_std_zero:10,} "
            f"({(count_std_zero/total_rows):.1%})"
        )
        logger.info(
            f"Instances with mean==min==max:  {count_equal_stats:10,} "
            f"({(count_equal_stats/total_rows):.1%})"
        )
        logger.info("=" * 40)        

        # Store the final DataFrame
        self.device_level_df = full_df
        self.device_level_df_temporal_feature_names = [
            c for c in full_df.columns if c not in ('bucket_idx', 'device_uri_str', 'property_type')
        ]
        
        return None
    
    ##############################
    # Cleanup
    ##############################
    
    @staticmethod
    def clean_anomalies(
            df:                 pd.DataFrame, 
            period:             int = 288, 
            offset:             int = 287,
            extra_buckets:      Optional[List[int]] = None,
            clear_high_counts:  bool = False
    ) -> pd.DataFrame:
        """
        Clean anomalies in device-level data step by step:

        1. Clear regular weekly spike buckets (periodic dumps).
        2. Clear user-specified anomalous buckets.
        3. Optionally clear all rows with count > 1 (rogue measurements).

        Args:
            df: device_level_df with 'bucket_idx', 'count', 'has_measurement' and stat columns.
            period: interval of spike repetition (e.g. 288 for 30min buckets = 7 days).
            offset: bucket offset where spikes occur (e.g. 287 = Saturday 23:30 if start=0).
            extra_buckets: list of additional bucket_idx values to clean.
            clear_high_counts: if True, clears out all rows where count > 1.

        Returns:
            Modified DataFrame with anomalies NaN'ed out, count=0, has_measurement=0.
        """
        df = df.copy()
        
        # =============== 1) Regular weekly spike buckets ===============
        weekly_mask = (df["bucket_idx"] % period == offset)
        n_weekly = weekly_mask.sum()
        if n_weekly > 0:
            df.loc[weekly_mask, "count"] = 0.0
            df.loc[weekly_mask, "has_measurement"] = 0.0
            for col in ["mean", "std", "max", "min"]:
                if col in df.columns:
                    df.loc[weekly_mask, col] = np.nan
        logger.info(f"[Step 1] Cleared {n_weekly:,} rows from regular weekly spikes (offset={offset}, period={period}).")
        
        # =============== 2) Extra manually specified buckets ===============
        extra_mask = pd.Series(False, index=df.index)
        if extra_buckets is not None and len(extra_buckets) > 0:
            extra_mask = df["bucket_idx"].isin(extra_buckets)
            n_extra = extra_mask.sum()
            if n_extra > 0:
                df.loc[extra_mask, "count"] = 0.0
                df.loc[extra_mask, "has_measurement"] = 0.0
                for col in ["mean", "std", "max", "min"]:
                    if col in df.columns:
                        df.loc[extra_mask, col] = np.nan
            logger.info(f"[Step 2] Cleared {n_extra:,} rows from extra buckets {extra_buckets}.")
        else:
            logger.info("[Step 2] No extra buckets specified.")
        
        # =============== 3) Clear rows with count > 1 ===============
        n_high_counts = 0
        if clear_high_counts:
            high_mask = (df["count"] > 1)
            n_high_counts = high_mask.sum()
            if n_high_counts > 0:
                df.loc[high_mask, "has_measurement"] = 0.0
                if "mean" in df.columns:
                    df.loc[high_mask, "mean"] = np.nan
            
            # Drop unused columns, as they are unneccessary when the "count < 2"
            drop_cols = [c for c in ["std", "max", "min", "count"] if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

            logger.info(f"[Step 3] Cleared {n_high_counts:,} rows where count > 1 "
                        f"and dropped columns {drop_cols}.")
        else:
            logger.info("[Step 3] clear_high_counts=False, skipped.")
        
        total_cleared = n_weekly + (extra_mask.sum() if extra_buckets else 0) + n_high_counts
        logger.info(f"=== Total cleaned rows: {total_cleared:,} ===")
        
        # =============== 4) Per-device-property NaN stats ===============
        if {"device_uri_str", "mean", "property_type"}.issubset(df.columns):
            nan_percents = (
                df.groupby(["device_uri_str", "property_type"])["mean"]
                .apply(lambda x: x.isna().mean() * 100)
                .sort_values(ascending=False)
            )
            logger.info("[Per-device-property NaN percentage after cleaning] (analyzed col: mean):")
            for (dev, prop), perc in nan_percents.items():
                logger.info(f"  {dev} ({prop}): {perc:.1f}% NaN")
                if perc > 75:
                    logger.info(f"(!) Device {dev} ({prop}) has >75% NaNs after cleaning. (!)")
                
        return df
    
    ##############################
    # Room-level DataFrame
    ##############################

    def build_room_level_df(self) -> None:
        """
        (--- outdated docstring ---)

        Transforms device-level data into a room-level feature matrix.

        This method pivots the `self.device_level_df` (keyed by device, property,
        and time bucket) into a wide-format DataFrame where each row represents a
        unique room and time bucket (`room_uri_str`, `bucket_idx`).

        For each property type (e.g., "Temperature"), it generates a set of features
        by aggregating data from all devices of that type within a single room.
        This creates distinct columns such as:
        - `Temperature_mean`, `Temperature_std`, `Temperature_max`, `Temperature_min`
        - `Temperature_count` (total number of measurements)
        - `Temperature_n_active_devices` (count of devices with measurements)
        - `Temperature_has_measurement` (a binary indicator: 1 if any device reported data, 0 otherwise)

        Aggregation Rules:
        - `mean`, `std`, `max`, `min`: The mean of the statistic across all active devices in the room.
        - `count`: The sum of measurement counts from all devices.
        - `has_measurement`:
            - `sum` -> `n_active_devices`: The number of devices reporting data.
            - `max` -> `has_measurement`: A binary flag indicating if the room had any data for that property.
        
        The final DataFrame is stored in `self.room_level_df`.
        """
        logger.info(f"Starting to build room_level_df...")
        
        # 1) Ensure device_level_df exists
        if not hasattr(self, 'device_level_df'):
            raise ValueError("device_level_df not found. Run build_device_level_df() first.")
        df = self.device_level_df.copy()
        
        
        # 2) Mapping each device_uri_str → room_uri_str
        df["room_uri_str"] = df['device_uri_str'].apply(self.office_graph._map_device_uri_str_to_room_uri_str)
        df['room_uri_str'] = df['room_uri_str'].astype('category')
        
        # ---------- Intermediate logging ----------
        unique_buckets = df['bucket_idx'].nunique()
        unique_devices = df['device_uri_str'].nunique()
        unique_rooms_mapped = df['room_uri_str'].nunique()

        logger.info(f"Mapped {unique_devices} unique devices "
                    f"into {unique_rooms_mapped} rooms ")
        logger.info(f"Total number of unique time buckets: {unique_buckets}.")
        # ---------- Intermediate logging ----------
        
        
        # 3) Using pivot_table to aggregate and pivot
        if 'count' in df.columns and 'std' in df.columns:
            logger.info("Performing full aggregation on detailed device-level stats.")
            values_to_agg = ['mean', 'std', 'max', 'min', 'count', 'has_measurement']
            agg_functions = {
                'mean': ['mean', 'std'],
                'std': 'mean', 
                'max': 'max', 
                'min': 'min',
                'count': 'sum', 
                'has_measurement': ['sum', 'max']
            }
        else:
            logger.info("Performing simplified aggregation on mean device-level values.")
            values_to_agg = ['mean', 'has_measurement']
            agg_functions = {
                'mean': ['mean', 'std', 'max', 'min'],
                'has_measurement': ['sum', 'max']
            }

        wide = pd.pivot_table(
            df,
            index=['room_uri_str', 'bucket_idx'],
            columns='property_type',
            values=values_to_agg,
            aggfunc=agg_functions
        )
        
        # Flatten the multi-level columns
        new_column_names = []
        for col in wide.columns:
            # col is always a 3-tuple: (value_name, aggfunc_name, prop_type)
            if isinstance(col, tuple) and len(col) == 3:
                value, aggfunc, prop = col
            else:
                raise ValueError(f"Unexpected column structure: {col}")
            
            # Special case: has_measurement
            if value == 'has_measurement': 
                if aggfunc=='sum': # sum → number of devices
                    name = f"{prop}_{'n_active_devices'}"
                else: # aggfunc=='max' → binary flag
                    name = f"{prop}_{'has_measurement'}"
            
            # Special case: mean
            elif value == 'mean': 
                # "{prop}_mean" or "{prop}_std"
                name = f"{prop}_{aggfunc}"
            
            # Special case: std
            elif value == 'std': 
                name = f"{prop}_average_intra_device_variation"
            
            # The standard case
            else:
                # max, min, count -> just "{prop}_{value}"
                name = f"{prop}_{value}"
            
            new_column_names.append(name)
        
        wide.columns = new_column_names
        wide = wide.reset_index()
        
        # Handling the standard deviation of means when only one device is active 
        property_types = self.device_level_df['property_type'].unique()
        
        for prop in property_types:
            inter_device_std_col = f"{prop}_std"
            n_active_col = f"{prop}_n_active_devices"

            if inter_device_std_col in wide.columns and n_active_col in wide.columns:
                # Condition: The inter-device std is NaN AND there was only 1 active device
                is_nan_mask = wide[inter_device_std_col].isnull()
                is_one_device_mask = wide[n_active_col] == 1

                # Apply the fix
                wide.loc[is_nan_mask & is_one_device_mask, inter_device_std_col] = 0.0
        
        # 4) Store the final resulting DataFrame
        self.room_level_df = wide
        
        logger.info(f"Built room_level_df DataFrame for measured rooms. Shape: {wide.shape}")
        
        return None
    
    def build_expanded_room_level_df(self) -> None:
        """
        Helper function to take the DataFrame of measured rooms and expand
        it to include all rooms defined in the graph, filling missing ones with NaNs.

        This is required for the STGCN pipeline, but do not call it for the Tabular pipeline.
        """
        if not hasattr(self, "room_level_df"):
            raise ValueError("Run build_room_level_df() first.")
        df = self.room_level_df.copy()

        logger.info("Expanding feature DataFrame to include all graph nodes for STGCN...")

        # 1. Get the complete set of all rooms and time buckets
        # These are the dimensions of our final grid.
        all_room_uri_strs = [str(uri) for uri in self.office_graph.rooms.keys()]
        all_bucket_indices = range(len(self.time_buckets))

        # 2. Create the full MultiIndex from the product of all rooms and all buckets
        # This represents every possible (room, bucket_idx) pair.
        full_grid_index = pd.MultiIndex.from_product(
            [all_room_uri_strs, all_bucket_indices],
            names=["room_uri_str", "bucket_idx"]
        )
        
        # 3. Re-index the existing DataFrame to this full grid
        expanded_df = (
            df
            .set_index(["room_uri_str", "bucket_idx"])   # make these two levels the index
            .reindex(full_grid_index)                   # add missing (room, bucket) pairs
            .reset_index()                              # restore columns 'room_uri_str' & 'bucket_idx'
        )
        expanded_df['room_uri_str'] = expanded_df['room_uri_str'].astype('str')
        expanded_df['room_uri_str'] = expanded_df['room_uri_str'].astype('category')

        # 4. For the newly added rows, some values can be safely imputed.
        for prop in self.used_property_types:
            # count
            count_col = f"{prop}_count"
            if count_col in expanded_df.columns:
                expanded_df[count_col] = expanded_df[count_col].fillna(0.0)
            
            # has_measurement
            has_meas_col = f"{prop}_has_measurement"
            if has_meas_col in expanded_df.columns:
                expanded_df[has_meas_col] = expanded_df[has_meas_col].fillna(0.0)
            
            # n_active_devices
            n_active_col = f"{prop}_n_active_devices"
            if n_active_col in expanded_df.columns:
                expanded_df[n_active_col] = expanded_df[n_active_col].fillna(0.0)

        # 5. Update the class attribute
        self.room_level_df_expanded = expanded_df

        n_rows, n_cols = expanded_df.shape
        logger.info(f"DataFrame expanded. New shape: {n_rows} rows, {n_cols} columns.")
        logger.info(f"Grid covers {len(all_room_uri_strs)} rooms × {len(all_bucket_indices)} buckets "
                    f"= {len(all_room_uri_strs) * len(all_bucket_indices)} theoretical rows.")
        
        return None

    def build_static_room_features_df(self) -> None:
        """
        Builds a DataFrame of static room features and stores it in the instance.
        
        This method iterates through all rooms in the graph, collects their static 
        attributes using the `_collect_room_static_feature_dict` helper, and
        stores the result in `self.static_room_features_df`.
        """
        logger.info(f"Building DataFrame with {len(self.static_room_attributes)} static features for all rooms...")
        
        # Create the DataFrame
        static_df = pd.DataFrame(
            [self._collect_room_static_feature_dict(str(uri))
            for uri in self.office_graph.rooms]
        ).set_index("room_uri_str")
        
        # Keep only selected attributes, preserve order
        static_df = static_df[self.static_room_attributes]
        
        # Ensure canonical room ordering (same as self.room_URIs_str)
        static_df = static_df.reindex([str(uri) for uri in self.room_URIs_str])
        
        # Make everything numeric
        # NOTE: Currently, there are no categorical features in the static room features,
        #       so we can safely convert all to numeric. But, if in the future they are
        #      added, this will need to be adjusted.
        static_df = static_df.apply(pd.to_numeric, errors="coerce")
        
        self.static_room_features_df = static_df
        logger.info(f"Successfully built and stored `static_room_features_df`. Shape: {self.static_room_features_df.shape}")
        
        return None

    def add_static_room_features_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches the df with static attributes for each room.

        This method creates a DataFrame of static room properties. 
        It then merges this static data into the main `room_level_df` based on the room's URI.

        Args:
            df: either the original or the expanded room_level_df DataFrame.
        """
        if not hasattr(self, 'static_room_features_df'):
            raise ValueError("Static room features DataFrame not found. Run build_static_room_features_df() first.")
        static_df = self.static_room_features_df.copy()

        merged_df = pd.merge(
            df.copy(), 
            static_df.reset_index(), 
            on="room_uri_str", 
            how="left"
        )
        
        logger.info("Successfully merged static room features.")
        return merged_df
    
    
    ##############################
    # Floor-level DataFrame
    ##############################

    def build_floor_level_df(self) -> None:
        """
        Builds a wide-format feature DataFrame with one row per time bucket.

        This function aggregates room-level data to the floor-level and then pivots
        the result to create a wide DataFrame where each row is a unique `bucket_idx`.
        Columns are structured as `F{floor}_{property}_{statistic}`.
        
        This is built from `room_level_df` and produces the kind of tabular,
        wide-format data suitable for certain time-series models.

        The final DataFrame is stored in `self.floor_level_df`.
        """
        if not hasattr(self, 'room_level_df'):
            raise ValueError("room_level_df not found; ensure build_room_level_df() was called first.")
        
        logger.info("Building wide-format, floor-aggregated DataFrame (one row per bucket)...")
        df = self.room_level_df.copy()

        # 1) Add floor_number mapping to the room-level data
        room_to_floor_map = {
            room_uri_str: self.office_graph._map_floor_uri_str_to_floor_number(
                self.office_graph._map_room_uri_str_to_floor_uri_str(room_uri_str)
            )
            for room_uri_str in df['room_uri_str'].unique()
        }
        df['floor_number'] = df['room_uri_str'].map(room_to_floor_map)
        df['floor_number'] = df['floor_number'].astype(int)

        # For later use
        property_types = self.used_property_types
        floor_numbers = df['floor_number'].unique()

        # 2) First, aggregate from rooms to floors using groupby.
        # This creates an intermediate "long" DataFrame (one row per floor per bucket).
        agg_spec = {}
        value_cols = [c for c in df.columns if c.split('_')[0] in property_types]
        
        for col in value_cols:
            if col.endswith('_count'):
                agg_spec[col] = 'sum'
            elif col.endswith('_n_active_devices'):
                agg_spec[col] = 'sum'
            elif col.endswith('_max'):
                agg_spec[col] = 'max'
            elif col.endswith('_min'):
                agg_spec[col] = 'min'

            elif col.endswith('_average_intra_device_variation'):
                agg_spec[col] = 'mean'

            elif col.endswith('_has_measurement'):
                agg_spec[col] = ['sum', 'max'] # sum -> n_active_rooms, max -> has_measurement flag

            elif col.endswith('_mean'):
                agg_spec[col] = ['mean', 'std']
            elif col.endswith('_std'):
                agg_spec[col] = 'mean'
            
            else:
                raise ValueError(f"Unexpected column name: {col}")

        long_floor_df = df.groupby(['floor_number', 'bucket_idx']).agg(agg_spec)

        # 3) Now, pivot the intermediate DataFrame to get the desired wide format
        wide = long_floor_df.pivot_table(
            index='bucket_idx',
            columns='floor_number'
        )
        
        # 4) Flatten the complex multi-level columns into the desired F{floor}_{prop}_{stat} format
        new_column_names = []
        for col in wide.columns:
            # Column is always a 3-tuple: (base_name, aggfunc, floor_number)
            # e.g., ('CO2Level_mean', 'mean', 7)
            if isinstance(col, tuple) and len(col) == 3:
                base_name, agg_func, floor = col
                prop, stat_suffix = base_name.split('_', 1)
                final_stat_name = ""
            else:
                raise ValueError(f"Unexpected column structure: {col}")
            
            # Special case: has_measurement
            if stat_suffix == 'has_measurement':
                if agg_func == 'sum':
                    final_stat_name = 'n_active_rooms'
                else: # agg_func == 'max'
                    final_stat_name = 'has_measurement'
                    
            # Special case: mean
            elif stat_suffix == 'mean':
                final_stat_name = agg_func

            # Special case: std
            elif stat_suffix == 'std': 
                final_stat_name = f"average_intra_room_variation"

            # The standard case
            else:
                final_stat_name = stat_suffix
            
            new_column_names.append(f"F{floor}_{prop}_{final_stat_name}")

        wide.columns = new_column_names
        wide = wide.reset_index()
        
        # 5) Fix std for inter-room variation when only one room is active on a floor
        for floor in floor_numbers:
            for prop in property_types:
                inter_room_std_col = f"F{floor}_{prop}_std"
                n_active_rooms_col = f"F{floor}_{prop}_n_active_rooms"
                
                if inter_room_std_col in wide.columns and n_active_rooms_col in wide.columns:
                    is_nan_mask = wide[inter_room_std_col].isnull()
                    is_one_room_mask = wide[n_active_rooms_col] == 1
                    wide.loc[is_nan_mask & is_one_room_mask, inter_room_std_col] = 0.0
        
        # 6) Store or return the final DataFrame
        self.floor_level_df = wide
        logger.info(f"Stored wide-format, floor-aggregated DataFrame. Shape: {wide.shape}")
        return None
    
    
    
    ##############################
    # Building-level DataFrame
    ##############################

    def build_building_level_df(self) -> None:
        """
        Aggregates floor-level data to create a building-level feature DataFrame.

        This method transforms the `self.floor_level_df` (which is wide on floors)
        into a condensed DataFrame where each row still represents a single time bucket,
        but the columns now represent aggregated statistics for the entire building.

        The final DataFrame is stored in `self.building_level_df`.
        """
        if not hasattr(self, 'floor_level_df'):
            raise ValueError("floor_level_df not found; ensure build_floor_level_df() was called first.")
        
        logger.info("Aggregating floor-level data into a single building-level DataFrame...")
        
        # Start with a copy of the floor-level data
        df = self.floor_level_df.copy()
        
        # Initialize the new building-level DataFrame with the same index
        building_df = pd.DataFrame(index=df.index)
        building_df['bucket_idx'] = df['bucket_idx']

        # Iterate over each property type ('Temperature', 'CO2Level', 'Humidity') to aggregate
        for prop in self.used_property_types:
                        
            # Sum up counts, active rooms, and active devices
            for stat in ['count', 'n_active_devices', 'n_active_rooms']:
                cols = df.filter(like=f"_{prop}_{stat}").columns
                building_df[f"{prop}_{stat}"] = df[cols].sum(axis=1)

            # Take the max of maxes and min of mins
            for stat in ['max', 'min']:
                cols = df.filter(like=f"_{prop}_{stat}").columns
                building_df[f"{prop}_{stat}"] = df[cols].agg(func=stat, axis=1)

            # Take the max and sum of has_measurement flags
            for stat in ['max', 'sum']:
                cols = df.filter(like=f"_{prop}_has_measurement").columns
                colname_suffix = "has_measurement" if stat == 'max' else "n_active_floors"
                building_df[f"{prop}_{colname_suffix}"] = df[cols].agg(func=stat, axis=1)
            
            # Take the mean and std of the mean
            for stat in ['mean', 'std']:
                cols = df.filter(like=f"_{prop}_mean").columns
                building_df[f"{prop}_{stat}"] = df[cols].agg(func=stat, axis=1)
                        
            # Take the mean of "average intra-device variation" and "average intra-room variation"
            for stat in ['average_intra_device_variation', 'average_intra_room_variation']:
                cols = df.filter(like=f"_{prop}_{stat}").columns
                building_df[f"{prop}_{stat}"] = df[cols].mean(axis=1)

            # From std columns, create "average_intra_floor_variation"
            std_cols = df.filter(like=f"_{prop}_std").columns
            building_df[f"{prop}_average_intra_floor_variation"] = df[std_cols].mean(axis=1)
        
        # Handling the standard deviation of means when only one floor is active 
        for prop in self.used_property_types:
            inter_floor_std_col = f"{prop}_std"
            n_active_col = f"{prop}_n_active_floors"
            
            if inter_floor_std_col in building_df.columns and n_active_col in building_df.columns:
                # Condition: The inter-device std is NaN AND there was only 1 active device
                is_nan_mask = building_df[inter_floor_std_col].isnull()
                is_one_device_mask = building_df[n_active_col] == 1
                
                # Apply the fix
                building_df.loc[is_nan_mask & is_one_device_mask, inter_floor_std_col] = 0.0
        
        # Store the final result
        self.building_level_df = building_df.reset_index(drop=True)
        
        logger.info(f"Successfully built building-level DataFrame. Shape: {self.building_level_df.shape}")
        
        return None

    ##############################
    # Additional features
    ##############################
    
    def add_time_features_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time-based features onto a DataFrame on bucket_idx."""
        if not hasattr(self, 'time_buckets'):
            raise ValueError("Time buckets not found. Call initialize_time_parameters() first.")
        if 'bucket_idx' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'bucket_idx' column.")
        
        logger.info("Adding cyclical time features (hour, day of week)...")
        
        # Create a mapping from bucket index to its start timestamp
        ts_map = {i: tb[0] for i, tb in enumerate(self.time_buckets)}
        timestamps = df['bucket_idx'].map(ts_map)
        
        # Ensure timestamps are in datetime format for feature extraction
        dt = pd.to_datetime(timestamps)
        
        # Hour of Day
        hours = dt.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        
        # Day of Week
        dows = dt.dt.dayofweek  # Monday=0, Sunday=6
        df['dow_sin'] = np.sin(2 * np.pi * dows / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dows / 7)
        
        logger.info("Successfully added time features.")
        return df
    
    def add_workhour_labels_to_df(
            self,
            df: pd.DataFrame,
            workhour_colname: str = "is_workhour",
        ) -> pd.DataFrame:
        """Add binary work-hour labels onto a DataFrame on bucket_idx."""
        if not hasattr(self, "workhour_labels_df"):
            raise AttributeError("workhour_labels_df not found. Run get_workhour_labels() first.")
        if 'bucket_idx' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'bucket_idx' column.")

        logger.info("Adding work‑hour labels...")

        workhour_map = (
            self.workhour_labels_df
            .set_index('bucket_idx')[workhour_colname]
        )
        # Vectorised assignment
        df[workhour_colname] = df['bucket_idx'].map(workhour_map)

        logger.info("Successfully added work‑hour labels.")
        return df
    
    def add_weather_features_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather features into a DataFrame on bucket_idx."""
        if 'bucket_idx' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'bucket_idx' column.")
        if not hasattr(self, 'weather_df'):
            raise ValueError("weather_df not found; call get_weather_data() first.")
        weather_df = self.weather_df.copy()
        
        # Merge on bucket_idx
        df_ = df.copy()
        merged = pd.merge(
            df_,
            weather_df,
            on='bucket_idx',
            how='left'
        )
        return merged
    
    def build_time_features_df(self) -> None:
        """
        Create a per-bucket table with cyclical hour/weekday encodings
        and the binary work-hour flag, then store it for reuse.

        Produces:
            self.time_features_df       # DataFrame indexed by bucket_idx
            self.time_feature_names # ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_workhour']
        """
        if not hasattr(self, "time_buckets"):
            raise AttributeError("time_buckets not initialised; call initialize_time_parameters() first.")
        if not hasattr(self, "workhour_labels_df"):
            raise AttributeError("workhour_labels_df not found; run get_workhour_labels() first.")

        # 1) make a scaffold with all bucket indices
        df = pd.DataFrame({"bucket_idx": range(len(self.time_buckets))})
        df = self.add_time_features_to_df(df)
        df = self.add_workhour_labels_to_df(df)
        self.time_feature_names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_workhour"]
        self.time_features_df = df.set_index("bucket_idx")[self.time_feature_names]

        logger.info(
            "Built time_features_df with %d buckets and features %s",
            len(self.time_features_df),
            self.time_feature_names,
        )
        return None