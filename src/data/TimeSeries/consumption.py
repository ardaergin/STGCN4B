import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def parse_consumption_filename(filename: str) -> datetime:
    """
    Parse date from consumption data filename.
    Expected format: meetdata_YYYY_MM_DD.csv
    
    Args:
        filename: Filename to parse
        
    Returns:
        datetime object representing the date
    """
    # Extract the base filename without path
    base_name = os.path.basename(filename)
    
    # Extract date parts from filename
    parts = base_name.replace("meetdata_", "").replace(".csv", "").split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid filename format: {filename}")
    
    # Convert parts to integers and create a datetime object
    try:
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        # Return a datetime object, not a string
        return datetime(year, month, day)
    except ValueError as e:
        # Add better error handling
        raise ValueError(f"Invalid date in filename {filename}: {str(e)}")

def load_consumption_files(
    consumption_dir: str, 
    start_time: datetime, 
    end_time: datetime
) -> Dict[datetime.date, pd.DataFrame]:
    """
    Load consumption CSV files within the specified date range.
    """
    logger.info(f"Loading consumption data from {consumption_dir}")
    
    # Convert string dates to datetime if needed
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        
    logger.info(f"Using date range: {start_time} to {end_time}")
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(consumption_dir, "meetdata_*.csv"))
    logger.info(f"Found {len(csv_files)} consumption files")
    
    # Filter files by date
    consumption_data = {}
    for file_path in csv_files:
        try:
            # Parsing file names using the helper function
            file_datetime = parse_consumption_filename(file_path)
            file_date = file_datetime.date()
            
            # Skip if outside date range
            if file_date < start_time.date() or file_date > end_time.date():
                continue
                
            # Load the CSV
            df = pd.read_csv(file_path)
            
            # Clean up and convert the data
            df = df.iloc[:, :2]  # Keep only first two columns
            df.columns = ['time', 'consumption']  # Rename columns
            
            # Convert 'mv' status rows if needed
            if df['consumption'].dtype == object:  # More robust type checking
                mask = df['consumption'] != 'mv'
                df = df[mask]
                df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
            
            # Store in dictionary with date as key
            consumption_data[file_date] = df
            
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(consumption_data)} days of consumption data")
    return consumption_data

def aggregate_consumption_to_time_buckets(
    consumption_data: Dict[datetime.date, pd.DataFrame],
    time_buckets: List[Tuple[datetime, datetime]],
    interval: str
) -> Dict[int, float]:
    """
    Aggregate consumption data to match time buckets.
    Maps the timestamp in consumption data to the end time of time buckets.
    
    Args:
        consumption_data: Dictionary of consumption DataFrames by date
        time_buckets: List of (start_time, end_time) tuples
        interval: Time interval ('15min', '30min', '1h', or '2h')

    Returns:
        Dictionary mapping time bucket index to consumption value
    """
    logger.info(f"Aggregating consumption data to match time buckets with interval {interval}")
    
    if interval not in ['15min', '30min', '1h', '2h']:
        raise ValueError(f"Unsupported interval: {interval}. Use '15min', '30min', '1h', or '2h'.")

    # Initialize result dictionary
    bucket_consumption = {}
    
    # Create a lookup from (date, end_time_str) to bucket index
    time_bucket_lookup = {}
    for i, (bucket_start, bucket_end) in enumerate(time_buckets):
        bucket_date = bucket_start.date()
        time_str = bucket_end.strftime('%H:%M')
        time_bucket_lookup[(bucket_date, time_str)] = i
    
    # Process each date's consumption data
    for date, df in consumption_data.items():

        if interval == '15min':
            sum_up_rows = 1
        elif interval == '30min':
            sum_up_rows = 2
        elif interval == '1h':
            sum_up_rows = 4
        elif interval == '2h':
            sum_up_rows = 8

        df_reset = df.reset_index(drop=True)
        groups = [df_reset.iloc[i:i+sum_up_rows] for i in range(0, len(df_reset), sum_up_rows)]
        interval_df = pd.DataFrame({
            'time': [group.iloc[-1]['time'] for group in groups],
            'consumption': [group['consumption'].mean() for group in groups]
        })

        for _, row in interval_df.iterrows():
            time_str = row['time']
            consumption_value = row['consumption']
                        
            # Look up the bucket index
            lookup_key = (date, time_str)
            if lookup_key in time_bucket_lookup:
                bucket_idx = time_bucket_lookup[lookup_key]
                bucket_consumption[bucket_idx] = consumption_value
    
    logger.info(f"Created consumption data for {len(bucket_consumption)} time buckets")

    sorted_bucket_consumption = {k: bucket_consumption[k] for k in sorted(bucket_consumption.keys())}

    return sorted_bucket_consumption
