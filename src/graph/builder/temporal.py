import numpy as np
from datetime import datetime
import pandas as pd
import logging
from rdflib import URIRef

# Logging setup
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class TemporalBuilderMixin:
    from ..officegraph import OfficeGraph
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
                "bucket_indices": list(bucket_list) # Use a copy
            }
        
        return None
    
    #############################
    # Weather Data (Predictor)
    #############################

    def get_weather_data(self,
                        weather_csv_path: str = "data/weather/hourly_weather_2022_2023.csv",
                        add_weather_code_onehot_features: bool = False
                        ) -> None:
        """
        Load, aggregate, and feature-engineer weather data for forecasting.

        Args:
            weather_csv_path: CSV with hourly weather (must have a 'date' column).

        Returns:
            Dict[bucket_idx → {feature_name: value}].
        """
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
            weather_df.drop(columns=["weather_code"], inplace=True)

        self.weather_features_ = weather_df
        self.weather_data_dict = weather_df.to_dict(orient="index")
        return None

    #############################
    # Task-Specific (Target) Data
    #############################
    
    def get_classification_labels(self, country_code: str = 'NL') -> None:
        """
        Generate work hour classification labels for each time bucket.
        
        Args:
            country_code: Country code for holidays
            
        Returns:
            Binary labels for each time bucket (1 for work hour, 0 for non-work hour), as np.ndarray.
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
        self.workhour_labels = labels

        logger.info(f"Generated {len(labels)} classification labels")
        logger.info(f"Work hours: {labels.sum()} ({labels.sum()/len(labels):.1%} of time buckets)")
        logger.info("Workhour labels are saved as an array to 'self.workhour_labels'.")

        return None
    
    def get_forecasting_values(self, consumption_dir: str = "data/consumption") -> None:
        """
        Load and aggregate consumption data for forecasting.

        Args:
            consumption_dir: Directory containing consumption data files

        Returns:
            NumPy array of consumption values, one per time bucket.
        """        
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

        # 2) Convert to array format
        T = len(self.time_buckets)
        consumption_array = np.array([bucket_consumption[i] for i in range(T)], dtype=float).reshape(-1, 1)
        final_consumption_array = consumption_array.flatten() # Ensure 1D

        # 3) Store as a class attribute (as an array)
        self.consumption_values = final_consumption_array
        logger.info("Consumption values are saved as an array to 'self.consumption_values'.")

        return None
        
    #############################
    # Measurement bucketing
    #############################

    def bucket_measurements_by_device_property(self) -> None:
        """
        Bucket measurements by device and property according to time buckets,
        using a single pass and pandas groupby for speed.
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

        # Raise error if the DataFrame is empty
        df = pd.DataFrame.from_records(records)
        if df.empty:
            raise ValueError("No measurements to bucket.")
        
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
        if "Contact" in self.used_property_types:
            stats = ["mean", "std", "max", "min", "sum", "count"]
        else:
            stats = ["mean", "std", "max", "min", "count"]
        
        agg_df = (
            df.groupby(["device_uri","property_type","bucket_idx"])["value"]
            .agg(stats)
            .reset_index()
        )

        # Null‐out stats for 'Contact'
        if "Contact" in self.used_property_types:
            contact_mask = agg_df["property_type"] == "Contact"
            agg_df.loc[contact_mask, ["mean","std","max","min"]] = np.nan
        else:
            stats = ["mean", "std", "max", "min", "count"]

        # 4) Pick columns and save df
        cols = ["device_uri","property_type","bucket_idx"] + stats
        self.bucketed_measurements_df = agg_df[cols]

        # --- Sanity logging ---
        logger.info("-" * 40)
        logger.info(f"BUCKETING SUMMARY ({len(agg_df)} rows total)\n")

        # Unique vs expected
        n_buckets = agg_df['bucket_idx'].nunique()
        expected = len(self.time_buckets)
        logger.info(f"• Buckets covered: {n_buckets}, {expected}\n")

        # Missing buckets
        all_bins = set(range(expected))
        present = set(agg_df['bucket_idx'].unique())
        missing = sorted(all_bins - present)
        logger.info(f"• Number of missing buckets: {len(missing)}\n")

        # distribution of records per bucket
        counts = agg_df.groupby('bucket_idx').size()
        desc = counts.describe()
        logger.info(
            f"Per-bucket record counts summary: count={desc['count']}, mean={desc['mean']:.2f}, "
            f"std={desc['std']:.2f}, min={desc['min']}, 25%={desc['25%']}, "
            f"50%={desc['50%']}, 75%={desc['75%']}, max={desc['max']}"
        )

        # Per‐device stats
        logger.info("ROWS PER DEVICE:")
        for dev, cnt in agg_df['device_uri'].value_counts().items():
            ub = agg_df[agg_df['device_uri']==dev]['bucket_idx'].nunique()
            logger.info(f"  - {dev}: {cnt} rows in {ub} buckets")
        logger.info("")

        # Per‐property stats
        logger.info("ROWS PER PROPERTY:")
        for prop, cnt in agg_df['property_type'].value_counts().items():
            ub = agg_df[agg_df['property_type']==prop]['bucket_idx'].nunique()
            logger.info(f"  - {prop}: {cnt} rows in {ub} buckets")

        # sample row
        sample = agg_df.iloc[0].to_dict()
        logger.info(f"Sample row: {sample}")

        logger.info("-" * 40)
        return None
    
    #############################
    # Adding back the empty buckets
    #############################

    def build_full_feature_df(self) -> None:
        """
        Re-index the bucketed measurements so we get one row per (device_uri, property_type, bucket_idx), 
        and adding a binary 'has_measurement' indicator.

        IMPORTANT:  There is no normalization or missing value imputation by this point.
                    Will do this in the next steps.
        
        Returns:
            DataFrame with columns:
            ['device_uri', 'property_type', 'bucket_idx', <feature_cols>, 'has_measurement']
        """
        if not hasattr(self, "bucketed_measurements_df"):
            raise ValueError("Call normalize_bucketed_measurements first.")

        # 1) Tag existing normalized rows as present
        df = self.bucketed_measurements_df.copy()
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

        # 4) Fill NaNs with 0.0 for the 'has_measurement' and 'count' columns 
        # These imputations can be made, since we know the rows we just added 
        # - do not have a measurement (has_measurement = 0), 
        # - and, therefore, also has no count (count = 0)
        full_df['has_measurement'] = full_df['has_measurement'].fillna(0.0)
        full_df['count'] = full_df['count'].fillna(0.0)

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
    
    def build_room_feature_df(self) -> None:
        """
        Transforms device-level data into a room-level feature matrix.

        This method pivots the `self.full_feature_df` (keyed by device, property,
        and time bucket) into a wide-format DataFrame where each row represents a
        unique room and time bucket (`room_uri`, `bucket_idx`).

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

        The final DataFrame is stored in `self.room_feature_df`.
        """
        # 1) Ensure full_feature_df exists
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("full_feature_df not found. Run build_full_feature_df() first.")
        df = self.full_feature_df.copy()
        
        logger.info(f"Starting build_room_feature_df with {len(df)} rows in full_feature_df.")

        # 2) Map each device_uri → room_uri
        #    (device_uri is a string, so convert to URIRef and look up device_obj.room)
        def _map_device_to_room(dev_str: str):
            try:
                dev_ref = URIRef(dev_str)
                device_obj = self.office_graph.devices.get(dev_ref)
                if device_obj is None:
                    raise KeyError(f"Device {dev_str} not in office_graph.devices")
                return device_obj.room
            except Exception as e:
                raise RuntimeError(f"Cannot map device_uri '{dev_str}' to a room: {e}")

        df['room_uri'] = df['device_uri'].apply(_map_device_to_room)
        unique_devices = df['device_uri'].nunique()
        unique_rooms_mapped = df['room_uri'].nunique()
        logger.info(f"Mapped {unique_devices} unique device_uris into {unique_rooms_mapped} distinct rooms.")

        # Using pivot_table to aggregate and pivot
        wide = pd.pivot_table(
            df,
            index=['room_uri', 'bucket_idx'],
            columns='property_type',
            values=['mean', 'std', 'max', 'min', 'count', 'has_measurement'],
            aggfunc={
                'mean': 'mean', 
                'std': 'mean', 
                'max': 'mean', 
                'min': 'mean',
                'count': 'sum', 
                'has_measurement': ['sum', 'max']
            }
        )

        # Flatten the multi-level columns
        # col is a tuple, e.g., ('mean', 'Temperature') or ('has_measurement', 'sum', 'Temperature')
        new_cols = []
        for col in wide.columns:
            if col[0] == 'has_measurement': # The special case
                stat = col[1]  # 'sum' or 'max'
                prop = col[2]  # 'Temperature'
                stat_name = 'n_devices' if stat == 'sum' else 'has_measurement'
                new_cols.append(f"{prop}_{stat_name}")
            else: # The standard case
                stat = col[0]
                prop = col[1]
                new_cols.append(f"{prop}_{stat}")
        wide.columns = new_cols

        wide = wide.reset_index()

        # Store the resulting DataFrame
        self.room_feature_df = wide

        # Informative logging about DataFrame structure
        logger.info(f"Built room_feature_df DataFrame for measured rooms. Shape: {wide.shape}")

        num_unique_rooms = wide['room_uri'].nunique()
        num_unique_buckets = wide['bucket_idx'].nunique()
        total_columns = wide.shape[1]
        column_names = wide.columns.tolist()

        logger.info(f"room_feature_df contains {num_unique_rooms} unique rooms and {num_unique_buckets} unique buckets.")
        logger.info(f"room_feature_df has {total_columns} columns. Column names:")
        for col in column_names:
            logger.info(f"  • {col}")

        return None