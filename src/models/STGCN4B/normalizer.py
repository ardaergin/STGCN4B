from abc import ABC, abstractmethod
from typing import Union, List, Dict
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, RobustScaler

import logging; logger = logging.getLogger(__name__)


class STGCNNormalizer(ABC):
    """
    An abstract base class for data normalization in STGCN models.
    
    It defines a common interface for feature normalization and provides a concrete
    implementation for target normalization, which is shared across different
    graph types.
    """
    def __init__(self):
        self.target_scaler = None
        self.feature_scalers = None
    
    @staticmethod
    def _get_scaler(method: str):
        """
        Factory function that returns a new scaler instance based on the method.
        
        Args:
            method (str): 
                - 'mean' for StandardScaler, 
                - 'median' for RobustScaler.
        
        Returns:
            An un-fitted sklearn scaler instance.
        """
        if method == "mean":
            return StandardScaler(with_mean=True, with_std=True)
        elif method == "median":
            return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))
        else:
            raise ValueError(f"Unknown scaling method: {method}. Choose 'mean' or 'median'.")
    
    @abstractmethod
    def fit_features(self):
        """Abstract method to calculate feature statistics from training data."""
        pass
    
    @abstractmethod
    def transform_features(self):
        """Abstract method to apply feature transformation."""
        pass
    
    def fit_target(
            self, 
            y_train:        np.ndarray, 
            y_train_mask:   np.ndarray = None,
            method:         str = 'median'
    ) -> "STGCNNormalizer":
        """
        Args:
            y_train (np.ndarray): Training target array.
            train_mask (np.ndarray, optional): Mask to select valid targets.
            method (str): The scaling method to use: 'mean' or 'median'.
        """
        if y_train_mask is not None:
            valid_targets = y_train[y_train_mask.astype(bool)]
        else:
            valid_targets = y_train
        if valid_targets.size == 0:
            raise ValueError("No valid targets found for fitting the scaler.")
        
        self.target_scaler = self._get_scaler(method=method)
        self.target_scaler.fit(valid_targets.reshape(-1, 1))
        
        logger.info(f"Fitted target scaler using method={method}.")
        return self
    
    def transform_target(
            self, 
            targets: np.ndarray
    ) -> np.ndarray:
        """Normalizes the target array using the fitted target scaler."""
        if self.target_scaler is None:
            raise RuntimeError("Must call fit_target() before transforming.")
        return self.target_scaler.transform(targets.reshape(-1, 1)).reshape(-1)
    
    def inverse_transform_target(
            self, 
            predictions: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inverse-transforms predictions back to the original scale. This is crucial for evaluation."""
        if self.target_scaler is None:
            raise RuntimeError("Must call fit_target() before inverse transforming.")
        return self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(-1)


class Homogeneous(STGCNNormalizer):
    """
    Normalizer for homogeneous graph data.
    
    Notes:
    - The input features are a 3D tensor: (T, R, F), F is the feature dimension.
    - Data contains NaNs, zero-imputation is handled later.
    - Data is on CPU, batches are loaded to GPU later in training.
    """
    
    def __init__(self):
        super().__init__()
        self.feature_scalers: List[Union[StandardScaler, RobustScaler]] = []
        
    def fit_features(
            self, 
            train_data: np.ndarray, 
            feature_names: List[str], 
            method: str = 'median',
            features_to_skip_norm: List[str] = None
    ) -> "Homogeneous":
        """
        Calculates feature statistics from the training data slice.

        Args:
            train_data (np.ndarray): A 3D array of shape (T_train, R, F).
            feature_names (List[str]): A list of length F with feature names.
            method (str): The scaling method to use: 'mean' for Z-score or 'median' for Robust Scaling.
            features_to_skip_norm (List[str], optional): List of substrings for features to skip.
        """
        logger.info("Fitting homogeneous normalizer.")
                
        # Identify and handle features to skip (this logic is the same)
        self.feature_scalers = []
        for i, fname in enumerate(feature_names):
            if features_to_skip_norm and any(s in fname for s in features_to_skip_norm):
                logger.info(f"Skipping normalization for feature '{fname}'")
                self.feature_scalers.append(None)  # identity transform
            else:
                scaler = self._get_scaler(method=method)
                # collapse T,R into one axis
                vals = train_data[..., i].reshape(-1, 1)
                scaler.fit(vals)
                self.feature_scalers.append(scaler)
                
        logger.info(f"Fitted feature processor using method={method}.")
        return self
    
    def transform_features(
            self, 
            all_data: np.ndarray
    ) -> np.ndarray:
        """Applies transformation to the full (T, R, F) numpy array."""
        if not self.feature_scalers:
            raise RuntimeError("Must call fit_features() before transforming.")
        out = all_data.copy()
        for i, scaler in enumerate(self.feature_scalers):
            if scaler is not None:
                vals = out[..., i].reshape(-1, 1)
                out[..., i] = scaler.transform(vals).reshape(out[..., i].shape)
        return out


class Heterogeneous(STGCNNormalizer):
    """
    Normalizer for heterogeneous graph data.

    Notes:
    - The input features are HeteroData snapshots: Dict[int, HeteroData].
    - Data contains NaNs, zero-imputation is handled later.
    - Data is on CPU, batches are loaded to GPU later in training.
    """
    
    def __init__(self):
        super().__init__()
        self.feature_scalers: Dict[str, List[Union[StandardScaler, RobustScaler]]] = {}
        
    def fit_features(
            self,
            train_data: Dict[int, HeteroData],
            feature_names: Dict[str, List[str]],
            method: str = "median",
            features_to_skip_norm: List[str] | None = None,
    ) -> "Heterogeneous":
        logger.info("Fitting heterogeneous normalizer (no NaN imputation).")
        
        # 1. Collect node‑wise feature matrices
        collector: Dict[str, List[torch.Tensor]] = {nt: [] for nt in feature_names}
        for snap in train_data.values():
            for nt in snap.node_types:
                if "x" in snap[nt]:
                    collector[nt].append(snap[nt].x.float().cpu().numpy())
        
        # 2. Fit & Skip‑normalisation handling
        for nt, tensors in collector.items():
            if not tensors:
                continue
            joined = np.concatenate(tensors, axis=0)  # (N_total, C)
            names = feature_names.get(nt, [])
            self.feature_scalers[nt] = []
            
            for i, fname in enumerate(names):
                if features_to_skip_norm and any(s in fname for s in features_to_skip_norm):
                    logger.info(f"Node type '{nt}': skipping '{fname}'")
                    self.feature_scalers[nt].append(None)
                else:
                    scaler = self._get_scaler(method=method)
                    vals = joined[:, i].reshape(-1, 1)
                    scaler.fit(vals)
                    self.feature_scalers[nt].append(scaler)
        
        logger.info(f"Fitted feature processor using method={method}.")
        return self
    
    def transform_features(
            self, 
            all_data: Dict[int, HeteroData]
    ) -> Dict[int, HeteroData]:
        """Applies the transformation to all snapshots in-place."""
        if not self.feature_scalers:
            raise RuntimeError("Must call fit_features() before transforming.")
        
        norm_data = {}
        for t, snapshot in all_data.items():
            norm_snapshot = snapshot.clone()
            for nt in norm_snapshot.node_types:
                if "x" in norm_snapshot[nt]:
                    arr = norm_snapshot[nt].x.cpu().numpy()
                    scalers = self.feature_scalers[nt]
                    for i, scaler in enumerate(scalers):
                        if scaler is not None:
                            vals = arr[:, i].reshape(-1, 1)
                            arr[:, i] = scaler.transform(vals).reshape(-1)
                    norm_snapshot[nt].x = torch.tensor(arr, device=norm_snapshot[nt].x.device, dtype=norm_snapshot[nt].x.dtype)
            norm_data[t] = norm_snapshot
        return norm_data