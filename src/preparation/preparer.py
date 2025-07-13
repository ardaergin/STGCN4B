import os
import torch
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from abc import ABC, abstractmethod

from ..utils.filename_util import get_data_filename
from .feature import BlockAwareFeatureEngineer
from .target import BlockAwareTargetEngineer

import logging
logger = logging.getLogger(__name__)


class BaseDataPreparer(ABC):
    """
    An abstract base class for preparing data for different models.
    It handles shared logic like loading data and memory optimization,
    while defining a contract for model-specific steps.
    """
    def __init__(self, args: Any):
        self.args = args
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.metadata: Dict[str, Any] = {}
        
        # Initializing the main product(s) of the class
        self.source_colname: str = ""
        self.target_source_df: pd.DataFrame = pd.DataFrame()
        self.target_colnames: List[str] = []
        self.input_dict: Dict[str, Any] = {}
    
    def get_input_dict(self) -> Dict[str, Any]:
        """
        Main orchestration method that prepares and returns the input dictionary.
        This is the primary public method for this class.
        It follows a fixed sequence of steps (the "template").
        """
        # Step 1: Load data and metadata from disk
        self._load_data_from_disk()
        
        # Step 2: Initialize block-aware engineers
        self._initialize_engineers()
        
        # Step 3: Prepare targets (subclass-specific logic)
        self._prepare_target()
        if not self.args.task_type == "workhour_classification":
            self._handle_nan_targets()
        
        # Step 4: Drop features if requested, before _prepare_features()
        self._drop_requested_columns()
        
        # Step 5: Prepare features (subclass-specific logic)
        self._prepare_features()
        
        # Step 6: Prepare the final model-specific input dictionary
        self._prepare_input_dict()
        
        return self.input_dict
    
    def _load_data_from_disk(self) -> None:
        """Loads the base DataFrame and metadata from disk."""
        # DataFrame
        data_fname_base = get_data_filename(file_type="dataframe", 
                                            task_type=self.args.task_type,
                                            model_family=self.args.model_family,
                                            interval=self.args.interval, 
                                            incorporate_weather=self.args.incorporate_weather)
        data_file_path = os.path.join(self.args.processed_data_dir, f"{data_fname_base}.parquet")
        logger.info(f"Loading parquet data from {data_file_path}")
        self.raw_df = pd.read_parquet(data_file_path)
        self.raw_df = self._reduce_mem_usage(self.raw_df)
        self.df = self.raw_df.copy()
        
        # MetaData
        metadata_fname_base = get_data_filename(file_type="metadata", 
                                                task_type=self.args.task_type,
                                                model_family=self.args.model_family,
                                                interval=self.args.interval, 
                                                incorporate_weather=self.args.incorporate_weather)
        metadata_file_path = os.path.join(self.args.processed_data_dir, f"{metadata_fname_base}.joblib")
        logger.info(f"Loading joblib metadata from {metadata_file_path}")
        self.metadata = joblib.load(metadata_file_path)
        
        # Targets for certain tasks
        if self.args.task_type == "workhour_classification":
            filename = f'target_workhour_{self.args.interval}.parquet'
            file_path = os.path.join(self.args.processed_data_dir, filename)
            logger.info(f"Loading workhour labels from: {file_path}")
            self.workhour_df = pd.read_parquet(file_path)
            
        elif self.args.task_type == "consumption_forecast":
            filename = f'target_consumption_{self.args.interval}.parquet'
            file_path = os.path.join(self.args.processed_data_dir, filename)
            logger.info(f"Loading consumption values from: {file_path}")
            self.consumption_df = pd.read_parquet(file_path)
        
    @staticmethod
    def _reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        if verbose: logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
                c_min, c_max = df[col].min(), df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                    else: df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose:
            logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
            logger.info(f"Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%")
        return df
    
    def _initialize_engineers(self) -> None:
        """Initializes the feature and target engineers."""
        self.target_engineer = BlockAwareTargetEngineer(self.metadata['blocks'])
        self.feature_engineer = BlockAwareFeatureEngineer(self.metadata['blocks'])
    
    def _drop_requested_columns(self) -> None:
        """Drop any column whose name contains a substring from args.features_to_drop."""
        if not self.args.features_to_drop:
            return
        
        substrings_to_drop = self.args.features_to_drop
        
        cols_to_drop = [col for col in self.df.columns 
                        if any(sub in col for sub in substrings_to_drop)]
        
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped {len(cols_to_drop)} columns containing substrings {substrings_to_drop}.")
            logger.debug(f"Dropped columns: {cols_to_drop}")
    
    @abstractmethod
    def _prepare_target(self) -> None:
        """
        Prepares the target columns. The common logic for identifying the source
        column and generating forecast targets is handled here.
        
        Subclasses should call this method via `super()._prepare_target()` and
        then add any model-specific target processing.
        """
        if self.args.task_type == "workhour_classification":
            self.target_colnames.append('target_workhour')
            self.target_source_df = pd.DataFrame()
            self.target_df = self.workhour_df.copy()
        else:
            # 1. Get the source column name based on the task
            if self.args.task_type == "measurement_forecast":
                self.source_colname = f'{self.args.measurement_variable}_{self.args.measurement_variable_stat}'
                self.target_source_df = self.df[['bucket_idx', 'room_uri_str', self.source_colname]].copy()
            else:  # consumption_forecast
                self.source_colname = 'consumption'
                self.target_source_df = self.consumption_df.copy() # just bucket_idx and consumption
                        
            # 2. Call the engineer to add target columns to the DataFrame
            target_df_with_source_col, self.target_colnames, = self.target_engineer.add_forecast_targets_to_df(
                task_type=self.args.task_type, data_frame=self.target_source_df,
                source_colname=self.source_colname, horizons=self.args.forecast_horizons,
                prediction_type=self.args.prediction_type
            )
            self.target_df = target_df_with_source_col.drop(columns=self.source_colname)
            logger.info(f"Prepared {len(self.target_colnames)} target columns from {self.source_colname}. "
                        f"New columns: {self.target_colnames}")
    
    @abstractmethod
    def _handle_nan_targets(self) -> None:
        """
        Abstract method to handle NaN targets after creating them.
        
        Kind of NaNs we have based on task type:
        - consumption_forecast:
            - end-of-block NaNs (due to grouping by block)
        - measurement_forecast:
            - end-of-block NaNs (due to grouping by block & room_uri_str)
            - Missing measurements in certain time buckets, resulting in NaNs in target columns
        """
        pass
    
    @abstractmethod
    def _prepare_features(self) -> None:
        """Abstract method for model-specific feature engineering."""
        pass
    
    @abstractmethod
    def _prepare_input_dict(self) -> None:
        """
        Abstract method for preparing the final, model-specific input_dict.
        This is the last step in the pipeline.
        """
        pass



class LGBMDataPreparer(BaseDataPreparer):
    """Prepares data for the LightGBM model."""
    
    def __init__(self, args: Any):
        super().__init__(args)
        assert self.args.model_family == "tabular", "LGBMDataPreparer only supports 'tabular' model_family."
        assert self.args.model == "LightGBM", "LGBMDataPreparer only supports LightGBM model."
    
    def _prepare_target(self) -> None:
        """Generates forecast targets for the LGBM model."""
        super()._prepare_target()

        # NOTE: For LGBM, we have the target columns and feature columns in one DataFrame
        #       1. It is just easier to slice for train/val/test splits
        #       2. For the LGBM model, there can be missing buckets, so there can be a mismatch
        #          between the target_df and the feature_df if we just use as they are.
        if self.args.task_type == "measurement_forecast": 
            self.df = pd.merge(self.df, self.target_df, on=['bucket_idx', 'room_uri_str'], how='left')
        else:
            self.df = pd.merge(self.df, self.target_df, on='bucket_idx', how='left')
    
    def _handle_nan_targets(self) -> None:
        """
        Handling NaN targets for the LGBM model.
        
        Method: After target preparation, any row with a NaN in one of the target columns will be dropped.
        """
        initial_rows = len(self.df)
        self.df.dropna(subset=self.target_colnames, inplace=True)
        final_rows = len(self.df)
        lost_rows = initial_rows - final_rows
        lost_percentage = (lost_rows / initial_rows) * 100
        
        logger.info(f"Dropped {lost_rows} rows with NaN targets. {final_rows} rows remaining.")
        logger.info(f"Lost {lost_percentage:.2f}% of the data due to NaN targets.")
        
        if self.df.empty:
            raise ValueError("DataFrame is empty after dropping NaN targets.")
    
    def _prepare_features(self) -> None:
        """
        Engineers features for the LGBM model.
        
        This is mainly for adding lag and moving average features for the consumption variable,
        if the task is consumption forecasting and the feature is not already requested to be dropped.
        """
        if self.args.task_type == "consumption_forecast" and "consumption" not in self.args.features_to_drop:
            logger.info("Engineering lag and moving average features for consumption...")
            
            self.df = self.feature_engineer.add_moving_average_features(
                windows=self.args.windows, shift_amount=self.args.shift_amount, 
                data_frame=self.df, cols=['consumption'],
                use_only_original_columns=True, extra_grouping_cols=None)
            
            self.df = self.feature_engineer.add_lag_features(
                lags=self.args.lags,
                data_frame=self.df, cols=['consumption'],
                use_only_original_columns=True, extra_grouping_cols=None)
    
    def _prepare_input_dict(self) -> None:
        self.input_dict = {
            "blocks": self.metadata["blocks"],
            "df": self.df,
            "source_colname": self.source_colname,
            "target_source_df": self.target_source_df,
            "target_colnames": self.target_colnames
        }


class STGCNDataPreparer(BaseDataPreparer):
    """Prepares data specifically for the STGCN model."""
    
    def __init__(self, args: Any):
        super().__init__(args)
        assert self.args.model_family == "graph", "STGCNDataPreparer only supports 'graph' model_family."
        assert self.args.model == "STGCN", "STGCNDataPreparer only supports STGCN model."
        self.target_data: Dict[str, Any] = {}
        self.feature_data: Dict[str, Any] = {}
    
    def _prepare_target(self) -> None:
        super()._prepare_target()
        if self.args.task_type == "workhour_classification":
            workhour_array = self.workhour_df.sort_values('bucket_idx')['target_workhour'].to_numpy()
            self.target_data["target_array"] = workhour_array
        else:
            # Using the helper to create the target array, mask array, and source array (if delta prediction)
            if self.args.task_type == "consumption_forecast":
                has_room_dimension = False
            else: # measurement_forecast
                has_room_dimension = True
            
            # Target array
            self.target_data["raw_target_array"] = self._pivot_df_to_numpy(
                df=self.target_df, 
                columns_to_pivot=self.target_colnames, 
                has_room_dimension=has_room_dimension)
            
            # Source array (for delta prediction)
            if self.args.prediction_type == "delta":
                self.target_data["raw_target_source_array"] = self._pivot_df_to_numpy(
                    df=self.target_source_df, 
                    columns_to_pivot=[self.source_colname],
                    has_room_dimension=has_room_dimension)
    
    def _handle_nan_targets(self) -> None:
        """
        Handling NaN targets for the STGCN model. 
        
        Method: Create a mask to be fed to the STGCN model, and fill the NaNs at the target with 0s.
        """
        # Create the mask from the raw array
        target_mask = ~np.isnan(self.target_data["raw_target_array"])
        self.target_data['target_mask'] = target_mask.astype(float)
        
        # Impute the NaNs to get the final, clean target array
        self.target_data['target_array'] = np.nan_to_num(self.target_data["raw_target_array"], nan=0.0)
        
        if self.args.prediction_type == "delta":
            # Impute the NaNs for the source values as well
            # NOTE: Since the target_mask is created based on the NaNs in the target array, 
            #       and that same mask will be used during evaluation to filter which points are compared, 
            #       the corresponding NaNs in the raw_target_source_array would be ignored. 
            #       But it does not hurt to impute NaNs just in case.
            raw_target_source_array = self.target_data['raw_target_source_array']
            self.target_data['target_source_array'] = np.nan_to_num(raw_target_source_array, nan=0.0)
            
    def _prepare_features(self) -> None:        
        # Identify feature columns (all numeric cols except identifiers and targets)
        identifier_cols = ['bucket_idx', 'block_id', 'room_uri_str']
        feature_cols = sorted([c for c in self.df.columns if c not in identifier_cols])
        
        # Save
        self.feature_data["feature_names"] = feature_cols
        self.feature_data["n_features"] = len(feature_cols)
        
        # Use the generic helper to create the feature array and a feature mask (which we might ignore)
        feature_array = self._pivot_df_to_numpy(self.df, columns_to_pivot=feature_cols)
        self.feature_data["feature_array"] = feature_array
    
    def _pivot_df_to_numpy(self,
                           df: pd.DataFrame, 
                           columns_to_pivot: List[str],
                           has_room_dimension=True
                           ) -> np.ndarray:
        """
        Helper to convert DataFrame columns to a model-ready NumPy array, preserving NaNs.
        
        This helper is crucial for creating the correct array shapes for different tasks. 
        The model's input features are always 3D (T, R, F), but the target shapes differ based on the prediction task. 
        This function handles that distinction.
        
        +------------------------+------------------------------------+-----------------+----------------------+
        | Task Name                 | What it Predicts                  | Target Shape    | has_room_dimension |
        +========================+====================================+=================+======================+
        | `measurement_forecast`    | Value for each room (e.g., temp)  | ``(T, R, H)``   | ``True``           |
        +------------------------+------------------------------------+-----------------+----------------------+
        | `consumption_forecast`    | Single value for the building     | ``(T, H)``      | ``False``          |
        +------------------------+------------------------------------+-----------------+----------------------+
        | `workhour_classification` | Single value for the floor        | ``(T, 1)``      | ``False``          |
        +------------------------------------------------------------------------------------------------------+
        * T: Time steps, R: Rooms, F: Features, H: Horizons
        
        Args:
            df (pd.DataFrame): The source DataFrame containing the data.
            columns_to_pivot (List[str]): A list of column names to be pivoted.
            has_room_dimension (bool): If True, pivots to a 3D array (T, R, F/H).
                                       If False, pivots to a 2D array (T, F/H).
        
        Returns:
            np.ndarray: The pivoted array with NaNs preserved.
        """
        logger.info(f"Pivoting {len(columns_to_pivot)} columns. Has Room Dimension: {has_room_dimension}")
        
        # Case for data WITH a room dimension (Features for all task types, targets for measurement_forecast)
        if has_room_dimension:
            room_order = self.metadata["rooms"]
            T = sum(len(v["bucket_indices"]) for v in self.metadata["blocks"].values())
            R = len(room_order)
            F = len(columns_to_pivot)
            
            full_index = pd.MultiIndex.from_product(
                [range(T), room_order], names=['bucket_idx', 'room_uri_str']
            )
            pivoted_df = df.set_index(['bucket_idx', 'room_uri_str'])[columns_to_pivot].reindex(full_index)
            
            np_array = pivoted_df.values.reshape(T, R, F)
            logger.info(f"Created 3D NumPy array of shape {np_array.shape}")
            return np_array
        
        # Case for data WITHOUT a room dimension (targets for consumption_forecast)
        else:
            total_timesteps = sum(len(v["bucket_indices"]) for v in self.metadata["blocks"].values())
            values_df = df[['bucket_idx'] + columns_to_pivot].groupby('bucket_idx').first()
            np_array = values_df.reindex(range(total_timesteps))[columns_to_pivot].values
            logger.info(f"Created 2D NumPy array of shape {np_array.shape}")
            return np_array
    
    def _prepare_input_dict(self):        
        # Device
        device = torch.device("cuda" if self.args.enable_cuda and torch.cuda.is_available() else "cpu")
        
        # Getting the requested adj, and already converting the obtained graph structures to tensors
        adj_matrix, masked_adj_dict, outside_adj = self._get_requested_adjacency_tensors(device=device)

        # For the downstream code, ensuring we always have an ndarray, never None:
        target_source_array = self.target_data.get("target_source_array")
        if target_source_array is None: 
            target_source_array = np.zeros_like(self.target_data["target_array"])
        target_mask = self.target_data.get("target_mask")
        if target_mask is None: 
            target_mask = np.ones_like(self.target_data["target_array"], dtype=float)
        
        # Creatign the input dict
        self.input_dict = {
            "device": device,
            
            # Data indices in block format
            "blocks": self.metadata["blocks"],
            "block_size": self.metadata["block_size"],
            
            # Graph structure
            "rooms": self.metadata["rooms"],
            "n_nodes": self.metadata["n_nodes"],
            
            # Adjacency
            "adjacency_matrix": adj_matrix,
            "masked_adjacency_matrices": masked_adj_dict,
            "outside_adjacency_vector": outside_adj,
            
            # Feature data
            "feature_array": self.feature_data["feature_array"],
            "feature_names": self.feature_data["feature_names"],
            "n_features": self.feature_data["n_features"],
            
            # Target
            "target_array": self.target_data["target_array"],
            "target_mask": target_mask,
            "target_source_array": target_source_array, # for delta prediction
        }
    
    def _get_requested_adjacency_tensors(
            self, device: torch.device
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        The loaded metadata includes both "binary" and "weighted" adjacency dictionaries.
        
        Both of the dictionaries have:
            1. "adjacency_matrix"
            2. "masked_adjacency_matrices"
            3. "outside_vector"
        
        We get the requested "binary" or "weighted" adjacency dictionary.
        Then, we convert them into tensors and return them.
        """
        requested_adjacency = self.args.adjacency_type
        adjacency_data = self.metadata[requested_adjacency]
        
        adj_matrix_tensor = torch.from_numpy(adjacency_data["adjacency_matrix"]).float().to(device)
        masked_adj_dict_tensor = {
            k: torch.from_numpy(v).float().to(device)
            for k, v in adjacency_data["masked_adjacency_matrices"].items()
            }
        
        if self.args.incorporate_weather:
            outside_adj_tensor = torch.from_numpy(adjacency_data["outside_adjacency_vector"]).float().to(device)
        else:
            outside_adj_tensor = None
        
        return adj_matrix_tensor, masked_adj_dict_tensor, outside_adj_tensor