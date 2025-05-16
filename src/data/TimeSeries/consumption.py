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

def aggregate_consumption_to_hourly(
    consumption_data: Dict[datetime.date, pd.DataFrame],
    time_buckets: List[Tuple[datetime, datetime]]
) -> Dict[int, float]:
    """
    Aggregate 15-minute consumption data to match time buckets,
    assuming exactly 96 readings per day (4 per hour) in sequential order.
    
    Args:
        consumption_data: Dictionary of consumption DataFrames by date
        time_buckets: List of (start_time, end_time) tuples
        
    Returns:
        Dictionary mapping time bucket index to consumption value
    """
    logger.info("Aggregating consumption data to match time buckets")
    
    # Initialize result dictionary
    bucket_consumption = {}
    
    # Keep track of which time bucket indices correspond to which date and hour
    date_hour_to_bucket_idx = {}
    for i, (bucket_start, bucket_end) in enumerate(time_buckets):
        bucket_date = bucket_start.date()
        bucket_hour = bucket_start.hour
        
        # Key is a tuple of (date, hour)
        date_hour_to_bucket_idx[(bucket_date, bucket_hour)] = i
    
    # Process each date with consumption data
    for date, df in consumption_data.items():
        # Ensure data is sorted by the original order
        df_sorted = df.reset_index(drop=True)
        
        # Check if we have the expected 96 entries (24 hours × 4 readings per hour)
        if len(df_sorted) != 96:
            logger.warning(f"Expected 96 readings for {date}, got {len(df_sorted)}. Will process anyway.")
        
        # Group readings into hourly buckets (4 readings per hour)
        # Each hour corresponds to indices: hour*4, hour*4+1, hour*4+2, hour*4+3
        for hour in range(24):
            # Indices for this hour (4 readings of 15 minutes each)
            start_idx = hour * 4
            end_idx = start_idx + 4
            
            # Make sure we don't go out of bounds
            if end_idx > len(df_sorted):
                logger.warning(f"Insufficient data for hour {hour} on {date}")
                break
            
            # Get the 4 readings for this hour
            hourly_readings = df_sorted.iloc[start_idx:end_idx]
            
            # Calculate average consumption for this hour
            avg_consumption = hourly_readings['consumption'].mean()
            
            # Find the corresponding bucket index
            bucket_key = (date, hour)
            if bucket_key in date_hour_to_bucket_idx:
                bucket_idx = date_hour_to_bucket_idx[bucket_key]
                bucket_consumption[bucket_idx] = avg_consumption
    
    logger.info(f"Created consumption data for {len(bucket_consumption)} time buckets")
    return bucket_consumption
