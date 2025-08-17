import os
import torch
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from torch_geometric.data import HeteroData

from ..utils.filename_util import get_data_filename
from .feature import BlockAwareFeatureEngineer
from .target import BlockAwareTargetEngineer

import logging; logger = logging.getLogger(__name__)


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
        self.target_dict: Dict[str, Any] = {}
        self.target_source_df: pd.DataFrame = pd.DataFrame()
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
        self._post_prepare_target()
        self._handle_nan_targets()
        if self.args.mask_workhours:
            self._mask_workhours()
        
        # Step 4: Drop features if requested, before _prepare_features()
        self._drop_requested_features()
        
        # Step 5: Prepare features (subclass-specific logic)
        self._prepare_features()
        
        # Step 6: Prepare the final model-specific input dictionary
        self._prepare_input_dict()
        
        return self.input_dict
    
    def _load_data_from_disk(self) -> None:
        """Loads the base DataFrame and metadata from disk."""
        # DataFrame
        data_fname_base = get_data_filename(
            file_type       = "dataframe", 
            interval        = self.args.interval, 
            weather_mode    = self.args.weather_mode,
            model_family    = self.args.model_family,
            task_type       = self.args.task_type,
        )
        data_file_path = os.path.join(self.args.processed_data_dir, f"{data_fname_base}.parquet")
        logger.info(f"Loading parquet data from {data_file_path}")
        self.raw_df = pd.read_parquet(data_file_path)
        self.raw_df = self._reduce_mem_usage(self.raw_df)
        self.df = self.raw_df.copy()
        
        # MetaData
        metadata_fname_base = get_data_filename(
            file_type       = "metadata", 
            interval        = self.args.interval, 
            weather_mode    = self.args.weather_mode,
        )
        metadata_file_path = os.path.join(self.args.processed_data_dir, f"{metadata_fname_base}.joblib")
        logger.info(f"Loading joblib metadata from {metadata_file_path}")
        self.metadata = joblib.load(metadata_file_path)
        logger.info(f"Metadata keys: {list(self.metadata.keys())}")
        
        # Workhour labels for masking
        if self.args.mask_workhours:
            workhour_labels_fname = f'workhours_{self.args.interval}.parquet'
            workhour_labels_path = os.path.join(self.args.processed_data_dir, workhour_labels_fname)
            logger.info(f"Loading workhour labels from: {workhour_labels_path}")
            self.workhour_labels_df = pd.read_parquet(workhour_labels_path)
        
        # Consumption target (if applicable)
        if self.args.task_type == "consumption_forecast":
            filename = f'target_consumption_{self.args.interval}.parquet'
            file_path = os.path.join(self.args.processed_data_dir, filename)
            logger.info(f"Loading consumption values from: {file_path}")
            self.consumption_df = pd.read_parquet(file_path)
    
    @staticmethod
    def _reduce_mem_usage(
            df: pd.DataFrame,
            verbose: bool = True
    ) -> pd.DataFrame:
        """
        Iterate through numeric columns and downcast to save memory safely.
        - Keeps datetimes/categories/objects untouched.
        - Preserves integer dtypes when NaNs are absent; otherwise leaves them (avoids int→float surprises).
        - Forces floats to float32 (good for both TF32 & LightGBM). Falls back to float64 if needed.
        - Uses deep=True for more accurate memory stats.
        """
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        if verbose:
            logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")

        for col in df.columns:
            if col in df.select_dtypes(include=['datetime64[ns]', 'category', 'object']).columns:
                continue

            col_type = df[col].dtype

            # Skip booleans
            if pd.api.types.is_bool_dtype(col_type):
                continue

            # Integers
            if pd.api.types.is_integer_dtype(col_type):
                # If there are NaNs, don't cast to smaller numpy ints (would upcast to float)
                if df[col].isna().any():
                    continue
                c_min, c_max = df[col].min(), df[col].max()
                # Try smallest fits
                for t in (np.int8, np.int16, np.int32):
                    if c_min >= np.iinfo(t).min and c_max <= np.iinfo(t).max:
                        df[col] = df[col].astype(t)
                        break
                # else keep as is (likely int64)
                continue

            # Floats
            if pd.api.types.is_float_dtype(col_type):
                c_min, c_max = df[col].min(), df[col].max()
                # Prefer float32 (works with TF32 & LightGBM); use 64 only if needed
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                continue

            # Anything else (e.g. nullable ints), just skip
            # (nullable Int64 will often be fine; converting safely needs extra logic)
            # pass

        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        if verbose:
            logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
            if start_mem > 0:
                logger.info(f"Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%")
        return df
    
    def _initialize_engineers(self) -> None:
        """Initializes the feature and target engineers."""
        self.target_engineer = BlockAwareTargetEngineer(self.metadata['blocks'])
        self.feature_engineer = BlockAwareFeatureEngineer(self.metadata['blocks'])
    
    def _prepare_target(self) -> None:
        """
        Prepares the target columns. The common logic for identifying the source
        column and generating forecast targets is handled here.
        
        Subclasses should call this method via `super()._prepare_target()` and
        then add any model-specific target processing.
        """
        # 1. Get the source column name based on the task
        if self.args.task_type == "measurement_forecast":
            self.source_colname = f'{self.args.measurement_variable}_{self.args.measurement_variable_stat}'
            self.target_source_df = self.df[['bucket_idx', 'room_uri_str', self.source_colname]].copy()
        else:  # consumption_forecast
            self.source_colname = 'consumption'
            self.target_source_df = self.consumption_df.copy() # just bucket_idx and consumption
                    
        # 2. Call the engineer to add target columns to the DataFrame
        self.target_dict = self.target_engineer.add_forecast_targets_to_df(
                task_type         = self.args.task_type,
                data_frame        = self.target_source_df,
                source_colname    = self.source_colname,
                horizons          = self.args.forecast_horizons,
                prediction_type   = self.args.prediction_type,
                get_workhour_mask = self.args.mask_workhours,
                workhour_df       = self.workhour_labels_df if self.args.mask_workhours else None,
                workhour_colname  = "is_workhour"           if self.args.mask_workhours else None,
            )
        logger.info(f"Prepared {len(self.target_dict['target_colnames'])} target columns from {self.source_colname}. "
                    f"New columns: {self.target_dict['target_colnames']}")
    
    @abstractmethod
    def _post_prepare_target(self) -> None:
        """Any additional processing that needs to be done after _prepare_target() is called."""
        pass
    
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
    def _mask_workhours(self) -> None:
        """Mask data based on whether it is a workhour or not."""
        pass

    @abstractmethod
    def _drop_requested_features(self) -> None:
        """Abstract class to drop any requested feature whose name contains a substring from args.features_to_drop."""
        pass
    
    def _drop_requested_columns(self) -> None:
        """Feature dropping method for any model that works with DataFrames."""
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


class TabularDataPreparer(BaseDataPreparer):
    """Prepares data for a Tabular model."""
    
    def __init__(self, args: Any):
        super().__init__(args)
        assert args.model_family == "tabular", "TabularDataPreparer only supports 'tabular' model_family."
        assert len(args.forecast_horizons)==1, "Tabular models only support single-horizon forecasting."
        self.id_cols = ['bucket_idx', 'room_uri_str'] if self.args.task_type == "measurement_forecast" else ['bucket_idx']
        
    def _post_prepare_target(self) -> None:
        """
        For tabular models, we have the target columns and feature columns in one DataFrame
        1. It is just easier to slice for train/val/test splits.
        2. Due to missing buckets, there can be a mismatch between the target_df and the feature_df.
        """
        self.df = pd.merge(self.df, self.target_dict["target_df"], on=self.id_cols, how='left')
    
    @staticmethod
    def _log_drop(before: int, after: int, reason: str) -> None:
        """Helper to log the number of rows dropped."""
        lost = before - after
        pct  = (lost / before * 100) if before else 0.0
        logger.info(f"Dropped {lost} rows due to {reason}. {after} rows remain ({pct:.2f}% loss).")
    
    def _handle_nan_targets(self) -> None:
        """
        Method: After target preparation, any row with a NaN in one of the target columns will be dropped.
        """
        before_nan = len(self.df)
        self.df.dropna(subset=self.target_dict["target_colnames"], inplace=True)
        after_nan = len(self.df)
        self._log_drop(before_nan, after_nan, "NaN targets")
        if after_nan == 0:
            raise ValueError("DataFrame empty after dropping NaN targets.")
        
    def _mask_workhours(self) -> None:
        """Merge the workhour mask DataFrame into the main DataFrame, mask, then drop the mask columns."""
        self.df = pd.merge(self.df, self.target_dict["workhour_mask_df"], on=self.id_cols, how='left')

        before_mask = len(self.df)
        self.df = self.df[self.df[self.target_dict["workhour_mask_colnames"]].all(axis=1)]
        after_mask = len(self.df)
        self._log_drop(before_mask, after_mask, "work-hour mask")
        if after_mask == 0:
            raise ValueError("DataFrame empty after applying work-hour mask.")
        
        # We don't need the mask columns anymore
        self.df.drop(columns=self.target_dict["workhour_mask_colnames"], inplace=True)
    
    def _drop_requested_features(self) -> None:
        self._drop_requested_columns()
    
    def _prepare_features(self) -> None:
        """        
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
            "blocks":               self.metadata["blocks"],
            "df":                   self.df,
            "source_colname":       self.source_colname,
            "target_source_df":     self.target_source_df,
            "target_colnames":      self.target_dict["target_colnames"],
        }


class STGCNDataPreparer(BaseDataPreparer):
    """Prepares data specifically for the STGCN model."""
    
    def __init__(self, args: Any):
        super().__init__(args)
        assert args.model_family == "graph", "STGCNDataPreparer only supports 'graph' model_family."
        assert args.model == "STGCN", "STGCNDataPreparer only supports 'STGCN' model."
        self.device: torch.device = torch.device(
            "cuda" if args.enable_cuda and torch.cuda.is_available() else "cpu")
        self.graph_dict: Dict[str, Any] = {}
        self.target_data: Dict[str, Any] = {}
        self.target_has_room_dimension: bool = (args.task_type == "measurement_forecast")
        self.feature_data: Dict[str, Any] = {}
    
    def _load_data_from_disk(self) -> None:
        """Extension to the base class."""
        super()._load_data_from_disk()
        self.graph_dict = self._get_requested_adjacency_tensors(device=self.device)
    
    def _get_requested_adjacency_tensors(
            self, 
            device: torch.device
    ) -> Dict[str, Any]:
        """
        The loaded metadata includes both "binary" and "weighted" adjacency dictionaries.
        
        Both of the dictionaries have:
        1. "room_URIs_str":                     List[str] (N)
        2. "n_nodes":                           int (N)
        3. "horizontal_adj_matrix":             np.ndarray (NxN)
        4. "vertical_adj_matrix":               np.ndarray (NxN)
        5. "full_adj_matrix":                   np.ndarray (NxN)
        6. "horizontal_masked_adj_matrices":    Dict[int, np.ndarray (NxN)]
        7. "vertical_masked_adj_matrices":      Dict[int, np.ndarray (NxN)]
        8. "full_masked_adj_matrices":          Dict[int, np.ndarray (NxN)]
        9. "outside_adj_vector":                np.ndarray (N)

        We get the requested "binary" or "weighted" adjacency dictionary.
        Then, we convert them into tensors and return them.
        """
        requested_adjacency = self.args.adjacency_type
        adjacency_data = self.metadata[requested_adjacency]
        
        # Single matrix tensors
        h_adj_mat_tensor = torch.from_numpy(adjacency_data["horizontal_adj_matrix"]).float().to(device)
        v_adj_mat_tensor = torch.from_numpy(adjacency_data["vertical_adj_matrix"]).float().to(device)
        f_adj_mat_tensor = torch.from_numpy(adjacency_data["full_adj_matrix"]).float().to(device)
        o_adj_vec_tensor = torch.from_numpy(adjacency_data["outside_adj_vector"]).float().to(device)
        
        # List of masked adjacency matrices as tensors
        h_masked_adj_mat_tensors = {
            k: torch.from_numpy(v).float().to(device)
            for k, v in adjacency_data["horizontal_masked_adj_matrices"].items()
        }
        v_masked_adj_mat_tensors = {
            k: torch.from_numpy(v).float().to(device)
            for k, v in adjacency_data["vertical_masked_adj_matrices"].items()
        }
        f_masked_adj_mat_tensors = {
            k: torch.from_numpy(v).float().to(device)
            for k, v in adjacency_data["full_masked_adj_matrices"].items()
        }
        
        # Non-tensor
        room_URIs_str = adjacency_data["room_URIs_str"]
        n_nodes = adjacency_data["n_nodes"]
        
        return {
            "room_URIs_str":                room_URIs_str,
            "n_nodes":                      n_nodes,
            "h_adj_mat_tensor":             h_adj_mat_tensor,
            "v_adj_mat_tensor":             v_adj_mat_tensor,
            "f_adj_mat_tensor":             f_adj_mat_tensor,
            "h_masked_adj_mat_tensors":     h_masked_adj_mat_tensors,
            "v_masked_adj_mat_tensors":     v_masked_adj_mat_tensors,
            "f_masked_adj_mat_tensors":     f_masked_adj_mat_tensors,
            "o_adj_vec_tensor":             o_adj_vec_tensor,
        }
    
    def _post_prepare_target(self) -> None:
        """
        For STGCN, we mainly use the _pivot_df_to_numpy helper 
        to create the target array, mask array, and source array (if delta prediction).
        """
        # Target array
        self.target_data["raw_target_array"] = self._pivot_df_to_numpy(
            df                      = self.target_dict["target_df"],
            columns_to_pivot        = self.target_dict["target_colnames"],
            has_room_dimension      = self.target_has_room_dimension,
        )
        
        # Source array (for delta prediction)
        if self.args.prediction_type == "delta":
            self.target_data["raw_target_source_array"] = self._pivot_df_to_numpy(
                df                  = self.target_source_df, 
                columns_to_pivot    = [self.source_colname],
                has_room_dimension  = self.target_has_room_dimension,
            )
    
    def _handle_nan_targets(self) -> None:
        """
        Note: any imputation must be done later after normalization, this is essentially pre-handling.

        Handling NaN targets for the STGCN model. 
        
        Method: Create a mask to be fed to the STGCN model.
        
        **Note**: 
        We impute NaNs to the source array (in delta forecast)!
        Since the target_mask is created based on the NaNs in the target array, 
        and that same mask will be used during evaluation to filter which points are compared, 
        the corresponding zero-imputed values in the target_source_array will be ignored anyways. 
        """
        target_nan_mask = ~np.isnan(self.target_data["raw_target_array"])
        self.target_data['target_mask'] = target_nan_mask.astype(np.float32, copy=False)
        
        # Leave the NaNs in the target array, we will impute later
        self.target_data["target_array"] = self.target_data["raw_target_array"].astype(np.float32, copy=False)
        
        # Impute the NaNs for the source values
        if self.args.prediction_type == "delta":
            self.target_data["target_source_array"] = np.nan_to_num(
                self.target_data["raw_target_source_array"], nan=0.0
            ).astype(np.float32, copy=False)
        else:
            # For the downstream code, ensuring we always have an ndarray, never None:
            self.target_data["target_source_array"] = np.zeros_like(
                self.target_data["target_array"]
            ).astype(np.float32, copy=False)
        
    def _mask_workhours(self) -> None:
        """Multiplying the existing mask array with the workhour mask array."""
        # Get the workhour mask by pivoting the workhour mask DataFrame
        workhour_mask = self._pivot_df_to_numpy(
            df                  = self.target_dict["workhour_mask_df"],
            columns_to_pivot    = self.target_dict["workhour_mask_colnames"],
            has_room_dimension  = self.target_has_room_dimension,
        )
        
        # Apply the mask to the target array
        self.target_data["target_mask"] *= workhour_mask
    
    def _pivot_df_to_numpy(
            self,
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
            room_order = self.graph_dict["room_URIs_str"]
            T = sum(len(v["bucket_indices"]) for v in self.metadata["blocks"].values())
            R = len(room_order)
            F = len(columns_to_pivot)
            
            full_index = pd.MultiIndex.from_product(
                [range(T), room_order], names=['bucket_idx', 'room_uri_str']
            )
            pivoted_df = df.set_index(['bucket_idx', 'room_uri_str'])[columns_to_pivot].reindex(full_index)
            
            np_array = pivoted_df.values.reshape(T, R, F)
            logger.info(f"Created 3D NumPy array of shape {np_array.shape}")
            return np_array.astype(np.float32, copy=False)
        
        # Case for data WITHOUT a room dimension (targets for consumption_forecast)
        else:
            total_timesteps = sum(len(v["bucket_indices"]) for v in self.metadata["blocks"].values())
            values_df = df[['bucket_idx'] + columns_to_pivot].groupby('bucket_idx').first()
            np_array = values_df.reindex(range(total_timesteps))[columns_to_pivot].values
            logger.info(f"Created 2D NumPy array of shape {np_array.shape}")
            return np_array.astype(np.float32, copy=False)

    def _prepare_input_dict(self):
        """Common ground for all STGCNDataPreparer subclasses. Subclasses add feature data."""
        self.input_dict = {
            "device":                       self.device,
            # Data indices in block format
            "blocks":                       self.metadata["blocks"],
            "block_size":                   self.metadata["block_size"],
            # Graph structure
            "room_URIs_str":                self.graph_dict["room_URIs_str"],
            "n_nodes":                      self.graph_dict["n_nodes"],
            # Adjacency matrix
            "h_adj_mat_tensor":             self.graph_dict["h_adj_mat_tensor"],
            "v_adj_mat_tensor":             self.graph_dict["v_adj_mat_tensor"],
            "f_adj_mat_tensor":             self.graph_dict["f_adj_mat_tensor"],
            # Masked adjacency matrices
            "h_masked_adj_mat_tensors":     self.graph_dict["h_masked_adj_mat_tensors"],
            "v_masked_adj_mat_tensors":     self.graph_dict["v_masked_adj_mat_tensors"],
            "f_masked_adj_mat_tensors":     self.graph_dict["f_masked_adj_mat_tensors"],
            # Outside adjacency
            "o_adj_vec_tensor":             self.graph_dict["o_adj_vec_tensor"],
            # Target
            "target_array":                 self.target_data["target_array"],
            "target_mask":                  self.target_data["target_mask"],
            "target_source_array":          self.target_data["target_source_array"],
        }




class Homogeneous(STGCNDataPreparer):
    
    def __init__(self, args: Any):
        super().__init__(args)
        assert args.graph_type == "homogeneous", "Homogeneous-STGCNDataPreparer only supports homogeneous graph type."
    
    def _drop_requested_features(self) -> None:
        self._drop_requested_columns()
    
    def _prepare_features(self) -> None:
        """
        Prepares features for the STGCN model.
        
        Note: The features are always 3D (T, R, F) for STGCN.
        """
        # Identify feature columns (all numeric cols except identifiers and targets)
        identifier_cols = ['bucket_idx', 'block_id', 'room_uri_str']
        feature_cols = sorted([c for c in self.df.columns if c not in identifier_cols])
        
        # Save
        self.feature_data["feature_names"] = feature_cols
        self.feature_data["n_features"] = len(feature_cols)
        
        # Use the generic helper to create the feature array and a feature mask (which we might ignore)
        feature_array = self._pivot_df_to_numpy(
            df                  = self.df, 
            columns_to_pivot    = feature_cols,
            has_room_dimension  = True,
        )
        self.feature_data["feature_array"] = feature_array
    
    def _prepare_input_dict(self):
        super()._prepare_input_dict()
        feature_dict = {
            # Feature data
            "feature_array":        self.feature_data["feature_array"], # np.ndarray (T, R, F)
            "feature_names":        self.feature_data["feature_names"], # List[str]
            "n_features":           self.feature_data["n_features"],    # int
        }
        self.input_dict |= feature_dict

class Heterogeneous(STGCNDataPreparer):

    def __init__(self, args: Any):
        super().__init__(args)
        assert args.graph_type == "heterogeneous", "Heterogeneous-STGCNDataPreparer only supports heterogeneous graph type."
        assert args.weather_mode == "feature", "Heterogeneous-STGCNDataPreparer should only be used with weather mode 'feature'."
        self.hetero_input: Dict[str, Any] = {}

    def _load_data_from_disk(self) -> None:
        """Extends STGCNDataPreparer._load_data_from_disk() to load the HeteroData snapshots."""
        super()._load_data_from_disk()

        # MetaData
        hetero_fname_base = get_data_filename(
            file_type       = "hetero_input", 
            interval        = self.args.interval, 
        )
        hetero_file_path = os.path.join(self.args.processed_data_dir, f"{hetero_fname_base}.joblib")
        logger.info(f"Loading heterogeneous graph data from {hetero_file_path}")
        # Loading to CPU. carrying them to GPU after DataLoader and at training:
        self.hetero_input = torch.load(hetero_file_path, map_location='cpu') 

    def _drop_requested_features(self) -> None:
        """
        Drop features from the HeteroData objects based on substrings.

        This method iterates through the base graph and all temporal snapshots.
        For each node type, it identifies features whose names contain any of the
        substrings in `self.args.features_to_drop`, removes them from the
        feature tensor (`.x`), and updates the corresponding feature name list.
        """
        if not self.args.features_to_drop:
            logger.info("No features requested to be dropped. Skipping.")
            return

        substrings_to_drop = self.args.features_to_drop
        logger.info(f"Attempting to drop features containing substrings: {substrings_to_drop}")

        # A helper function to process a single graph object (either base or temporal)
        def _process_graph(graph: 'HeteroData', feature_names_dict: Dict[str, List[str]]):
            for node_type in graph.node_types:
                # Check if this node type has features and a corresponding feature name list
                if 'x' not in graph[node_type] or node_type not in feature_names_dict:
                    continue

                original_feature_names = feature_names_dict[node_type]
                
                # Determine which feature indices to KEEP
                indices_to_keep = [
                    i for i, name in enumerate(original_feature_names)
                    if not any(sub in name for sub in substrings_to_drop)
                ]

                # If the number of features to keep is the same, no changes needed
                if len(indices_to_keep) == len(original_feature_names):
                    continue

                dropped_count = len(original_feature_names) - len(indices_to_keep)
                if dropped_count > 0:
                    logger.info(f"Node type '{node_type}': Dropping {dropped_count} features.")
                    
                    # 1. Slice the feature tensor to keep only the desired columns
                    graph[node_type].x = graph[node_type].x[:, indices_to_keep]

                    # 2. Update the feature names list in the central dictionary
                    feature_names_dict[node_type] = [original_feature_names[i] for i in indices_to_keep]

        # Process the base graph first
        logger.debug("Processing base_graph for feature dropping...")
        _process_graph(self.hetero_input["base_graph"], self.hetero_input["feature_names"])
        
        # Process all temporal graph snapshots
        logger.debug(f"Processing {len(self.hetero_input['temporal_graphs'])} temporal graphs for feature dropping...")
        for bucket_idx, temporal_graph in self.hetero_input["temporal_graphs"].items():
            # Note: The feature name dictionary is shared, so we pass the same one
            _process_graph(temporal_graph, self.hetero_input["feature_names"])
            
        logger.info("Finished dropping requested features from all heterogeneous graphs.")
    
    def _prepare_features(self) -> None:
        """HeteroData snapshot already has the features prepared. They are already nicely tensorized."""
        pass
    
    def _prepare_input_dict(self):
        super()._prepare_input_dict()
        hetero_feature_data = {
            "base_graph":               self.hetero_input["base_graph"],                # HeteroData
            "temporal_graphs":          self.hetero_input["temporal_graphs"],           # Dict[int, HeteroData]
            "node_mappings":            self.hetero_input["node_mappings"],             # Dict[str, Dict[Union[str, Tuple[str, str]], int]]
            "reverse_node_mappings":    self.hetero_input["reverse_node_mappings"],     # Dict[str, Dict[int, Union[str, Tuple[str, str]]]]
                                                                                        # NOTE: Tuple[str, str] is used for device-property pairs, 
                                                                                        #       the rest of the node types are single strings. 
            "feature_names":            self.hetero_input["feature_names"],             # Dict[str, List[str]]
            "property_types":           self.hetero_input.get("property_types", []),
        }
        self.input_dict |= hetero_feature_data