import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_weather_csv(
    csv_path: str,
    start_time: datetime,
    end_time: datetime
) -> pd.DataFrame:
    """
    Load hourly weather data from a single CSV and filter to the desired time range.

    Args:
        csv_path: Path to the weather CSV (must have a 'date' column).
        start_time: Start of desired range (inclusive).
        end_time: End of desired range (exclusive).

    Returns:
        DataFrame indexed by datetime, with one row per hour and all weather variables as columns.
    """
    logger.info(f"Loading weather data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"] )

    # Strip any timezone info
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    # Filter to the requested window
    mask = (df["date"] >= start_time) & (df["date"] < end_time)
    df = df.loc[mask].copy()

    df.set_index("date", inplace=True)
    logger.info(f"Loaded {len(df)} hourly records between {start_time} and {end_time}")
    return df


def get_weather_data_for_time_buckets(
    weather_df: pd.DataFrame,
    time_buckets: List[Tuple[datetime, datetime]],
    interval: str
) -> Dict[int, Dict[str, float]]:
    """
    Map hourly weather into arbitrary time buckets.

    For buckets smaller than or equal to 1h, the hourly value covering the bucket is repeated.
    For buckets larger than 1h, the mean of the constituent hourly values is taken.

    Args:
        weather_df: DataFrame with hourly index and weather columns.
        time_buckets: List of (start, end) tuples for each bucket.
        interval: String like '15min', '30min', '1h', '2h'.

    Returns:
        Dictionary mapping bucket index to a dict of aggregated weather values.
    """
    logger.info(f"Aggregating weather to {len(time_buckets)} buckets at {interval}")
    bucket_weather: Dict[int, Dict[str, float]] = {}
    offset = pd.Timedelta(interval)
    one_hour = pd.Timedelta("1h")

    for idx, (start, end) in enumerate(time_buckets):
        if offset <= one_hour:
            # repeat same hourly observation
            floor_hour = start.replace(minute=0, second=0, microsecond=0)
            if floor_hour in weather_df.index:
                vals = weather_df.loc[floor_hour].to_dict()
                bucket_weather[idx] = {k: float(v) for k, v in vals.items()}
        else:
            raise NotImplementedError("Only hourly and smaller intervals are supported for now.")
            # average over multiple hours
            # But need to handle the wind direction case differently (0 - 360 degrees)
            # and the weather code (categorical, not numeric)

    logger.info(f"Aggregated weather for {len(bucket_weather)} buckets")
    return bucket_weather
