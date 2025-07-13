import math
import pandas as pd
from typing import List, Dict, Tuple, Literal

import logging
logger = logging.getLogger(__name__)


class BlockAwareTargetEngineer:
    """Class for block-aware target engineering."""
    
    def __init__(self, blocks: Dict[int, Dict[str, List[int]]]):
        self.blocks = blocks
        self._block_map = {
            idx: blk
            for blk, info in self.blocks.items()
            for idx in info["bucket_indices"]
        }
    
    def _assign_block_id(self, df: pd.DataFrame) -> pd.Series:
        """Internal helper: map each bucket_idx to its block_id so we can group by block."""
        return df["bucket_idx"].map(self._block_map)
    
    def add_forecast_targets_to_df(self,
                                task_type: str,
                                data_frame: pd.DataFrame,
                                source_colname: str,
                                horizons: List[int],
                                prediction_type: Literal['absolute', 'delta'] = 'absolute',
                                forecast_type: Literal['point', 'range'] = 'point',
                                aggregation_type: Literal['mean', 'sum'] = 'mean',
                                min_periods_ratio: float = 0.5
                                ) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
        """
        Creates future forecast targets for single or multiple horizons.
        
        This method is block-aware and supports both 'point' and 'range' forecasts.
        
        Args:
            task_type (str): 'consumption_forecast' or 'measurement_forecast'.
            data_frame (pd.DataFrame): The input DataFrame.
            source_colname (str): The column name from which to generate the target.
            horizons (List[int]): A list of time steps into the future.
                - For 'point' forecast: [1, 4, 8] creates targets for t+1, t+4, t+8.
                - For 'range' forecast: [4, 8] creates targets for agg(t+1..t+4), agg(t+1..t+8).
            prediction_type (str): 'absolute' or 'delta'. Only 'absolute' is supported for 'range' forecasts.
            forecast_type (str): 'point' for single future values, or 'range' for aggregated values over a period.
            aggregation_type (str): How to aggregate for 'range' forecasts. Can be 'mean' or 'sum'.
        
        Returns:
            df (pd.DataFrame): DataFrame with new target columns.
            target_colnames (List[str]): Names of the created target columns.
            delta_to_absolute_map (Dict[str, str]): A map from delta target names to their absolute counterparts. # Add this line
        """
        ##### Validation #####
        # Argument checks
        if prediction_type not in ['absolute', 'delta']:
            raise ValueError("prediction_type must be 'absolute' or 'delta'.")
        if forecast_type not in ['point', 'range']:
            raise ValueError("forecast_type must be 'point' or 'range'.")
        if aggregation_type not in ['mean', 'sum']:
            raise ValueError("aggregation_type must be 'mean' or 'sum' for 'range' forecasts.")
        
        # Horizons
        if not all(h > 0 for h in horizons):
            raise ValueError("All horizons must be positive integers.")
        
        # Delta limitation
        if forecast_type == 'range' and prediction_type == 'delta':
            raise ValueError("Delta prediction ('delta') is not supported for 'range' forecasts.")
        
        # DataFrame
        if source_colname not in data_frame.columns:
            raise KeyError(f"Required source column '{source_colname}' not found in DataFrame.")
        grouping_cols = ['block_id']
        if task_type == "measurement_forecast":
            if "room_uri_str" not in data_frame.columns:
                raise KeyError("Task requires 'room_uri_str' for grouping, but it's not in the DataFrame.")
            grouping_cols.append('room_uri_str')
        
        logger.info(f"Creating {forecast_type} targets from '{source_colname}' for horizons {horizons}...")
        
        df = data_frame.copy()
        df['block_id'] = self._assign_block_id(df)
        
        # Sorting based on block_id + bucket_idx (+ room_uri_str, optionally)
        df.sort_values(['bucket_idx'] + grouping_cols, inplace=True)
        
        # Groupby
        grouped_data = df.groupby(grouping_cols)[source_colname]
        
        target_colnames = []
        if forecast_type == 'point':
            for h in horizons:
                future_values = grouped_data.shift(-h)
                
                target_col = f'target_h_{h}'
                target_colnames.append(target_col)
                
                if prediction_type == 'absolute':
                    df[target_col] = future_values
                
                else:  # prediction_type == 'delta'
                    df[target_col] = future_values - df[source_colname]
        
        elif forecast_type == 'range':
            for h in horizons:
                target_col = f'target_{aggregation_type}_h_1_to_{h}'
                target_colnames.append(target_col)                
                
                # Reverse the series, apply rolling, then reverse back to get a "forward" rolling window
                min_periods = math.ceil(h * min_periods_ratio)
                future_aggregate = grouped_data.rolling(window=h, min_periods=min_periods).agg(aggregation_type).shift(-(h))
                # NOTE:
                # 1. grouped_data.rolling(window=h): Creates a rolling window of size 'h'.
                #    For a value at time 't', it considers data from [t-h+1, ..., t].
                # 2. .agg(aggregation_type): Calculates the sum or mean over that window.
                # 3. .shift(-h): Shifts the entire result 'h' steps into the past.
                #    The result at time 't' now corresponds to the original window from [t+1, ..., t+h].
                
                df[target_col] = future_aggregate

        df.drop(columns='block_id', inplace=True)
        nan_count = df[target_colnames].isna().sum().sum()
        logger.info(f"Created {len(target_colnames)} target column(s). Found {nan_count} total NaN targets.")

        return df, target_colnames