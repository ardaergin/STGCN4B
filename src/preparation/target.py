import pandas as pd
from typing import List, Dict, Literal, Optional, Any

import logging; logger = logging.getLogger(__name__)


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
    
    def add_forecast_targets_to_df(
            self,
            task_type:          Literal['consumption_forecast', 
                                        'measurement_forecast'],
            data_frame:         pd.DataFrame,
            source_colname:     str,
            horizons:           List[int],
            prediction_type:    Literal['absolute', 'delta'],
            get_workhour_mask:  bool                            = False,
            workhour_df:        Optional[pd.DataFrame]          = None,
            workhour_colname:   str                             = "is_workhour",
    ) -> Dict[str, Any]:
        """
        Creates block-aware future forecast targets for single or multiple horizons.
        
        Args:
            task_type (str): 'consumption_forecast' or 'measurement_forecast'.
            data_frame (pd.DataFrame): The input DataFrame.
            source_colname (str): The column name from which to generate the target.
            horizons (List[int]): A list of time steps into the future.
            prediction_type (str): 'absolute' or 'delta'. Only 'absolute' is supported for 'range' forecasts.
            get_workhour_mask (bool): If True, create a work-hour mask for each horizon.
            workhour_df (pd.DataFrame, optional): DataFrame with work-hour labels. Required if get_workhour_mask is True.
            workhour_colname (str): The name of the boolean/int column in workhour_df.
        
        Returns:
            A dictionary containing the keys:
                - target_df (pd.DataFrame): DataFrame with grouping columns and target columns.
                - mask_df (pd.DataFrame): DataFrame with grouping columns and mask columns.
                - target_colnames (List[str]): Names of the created target columns.
                - mask_colnames (List[str]): Names of the created work-hour mask columns.
        """
        ##### Preflight setup and validations #####
        # Argument validation
        if prediction_type not in ['absolute', 'delta']:
            raise ValueError("prediction_type must be 'absolute' or 'delta'.")
        
        # Horizon validation
        if not all(h > 0 for h in horizons):
            raise ValueError("All horizons must be positive integers.")
        
        # DataFrame
        if source_colname not in data_frame.columns:
            raise KeyError(f"Required source column '{source_colname}' not found in DataFrame.")        
        grouping_cols = ['block_id']
        final_grouping_cols = ['bucket_idx']
        if task_type == "measurement_forecast":
            if "room_uri_str" not in data_frame.columns:
                raise KeyError("Task requires 'room_uri_str' for grouping, but it's not in the DataFrame.")
            grouping_cols.append('room_uri_str')
            final_grouping_cols.append('room_uri_str')
        
        df = data_frame.copy()
        
        # Workhour mask
        if get_workhour_mask:
            if workhour_df is None:
                raise ValueError("get_workhour_mask=True but workhour_df=None")
            if workhour_colname not in workhour_df.columns:
                raise KeyError(f"'{workhour_colname}' missing from workhour_df")
            df = df.merge(
                workhour_df[['bucket_idx', workhour_colname]],
                on='bucket_idx', how='left'
            )
        
        ##### Target (and mask) Creation #####
        logger.info(f"Creating targets from '{source_colname}' for horizons {horizons}...")
        
        # Sorting and grouping based on block_id + bucket_idx (+ room_uri_str, optionally)
        df['block_id'] = self._assign_block_id(df)
        df.sort_values(['bucket_idx'] + grouping_cols, inplace=True)
        grouped_df = df.groupby(grouping_cols)
        
        target_colnames, mask_colnames = [], []
        
        # Create shifted target columns for each horizon
        shifted_src_df = pd.concat(
            [grouped_df[source_colname]
             .shift(-h)
             .rename(f"target_h_{h}") for h in horizons],
            axis=1
        )
        if prediction_type == "delta":
            shifted_src_df = shifted_src_df.sub(df[source_colname], axis=0)
        shifted_src_df = shifted_src_df.astype("float32", copy=False)
        target_colnames = [f"target_h_{h}" for h in horizons]
        target_df = pd.concat([df[final_grouping_cols], shifted_src_df], axis=1)
        
        # Create shifted mask columns (if requested)
        if get_workhour_mask:
            shifted_mask_df = pd.concat(
                [grouped_df[workhour_colname]
                 .shift(-h)
                 .rename(f"mask_h_{h}") for h in horizons],
                axis=1
            ).fillna(0).astype("float32", copy=False)
            mask_colnames   = [f"mask_h_{h}" for h in horizons]
            mask_df = pd.concat([df[final_grouping_cols], shifted_mask_df], axis=1)
        else:
            mask_df = None
                
        # Logging
        target_df_nan_count = target_df[target_colnames].isna().sum().sum()
        logger.info(f"Created {len(target_colnames)} targets. NaNs: {target_df_nan_count}.")
        if get_workhour_mask:
            mask_df_nan_count = mask_df[mask_colnames].isna().sum().sum()
            logger.info(f"Created {len(mask_colnames)} work-hour mask columns. NaNs: {mask_df_nan_count}.")
        
        return {
            "target_df":                target_df,
            "target_colnames":          target_colnames,
            "workhour_mask_df":         mask_df,
            "workhour_mask_colnames":   mask_colnames,
        }