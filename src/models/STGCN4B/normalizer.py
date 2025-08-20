from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer
)

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
        
    @staticmethod
    def _get_scaler(method: str):
        """
        Factory function that returns a new scaler instance based on the method.
                
        Returns:
            An un-fitted sklearn scaler instance.
        """
        if method == "standard":
            return StandardScaler(with_mean=True, with_std=True)
        elif method == "robust":
            return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))
        elif method == "minmax":
            return MinMaxScaler(feature_range=(0, 1))
        elif method == "maxabs":
            return MaxAbsScaler()
        elif method == "quantile_uniform":
            return QuantileTransformer(n_quantiles=5000, output_distribution="uniform", subsample=500000, random_state=0)
        elif method == "quantile_normal":
            return QuantileTransformer(n_quantiles=5000, output_distribution="normal", subsample=500000, random_state=0)
        elif method == "power_yeojohnson":
            return PowerTransformer(method="yeo-johnson", standardize=True)
        elif method == "power_boxcox":
            return PowerTransformer(method="box-cox", standardize=True)
        else:
            raise ValueError(f"Unknown scaling method: {method}.")
    
    @staticmethod
    def apply_log1p_safe(vals: np.ndarray, feature_name: str, context: str = "") -> np.ndarray:
        """
        Safely apply log1p to an array, clipping negatives to 0 and logging if clipping occurs.
        
        Args:
            vals: Input values (reshaped column).
            feature_name: Name of the feature for logging.
            context: Extra string to indicate 'fit' or 'transform'.
            method: Optional name of the scaler used.

        Returns:
            Transformed values (with log1p).
        """
        neg_mask = vals < 0
        if np.any(neg_mask):
            n_neg = np.sum(neg_mask)
            min_val = vals.min()
            logger.warning(
                f"[{context}] Feature '{feature_name}': {n_neg} negatives detected "
                f"(min={min_val:.3f}), clipped to 0 before log1p."
            )
        vals = np.log1p(np.clip(vals, a_min=0, a_max=None))
        logger.info(f"[{context}] Feature '{feature_name}': log1p applied.")
        return vals
    
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
            y_train_mask:   np.ndarray      = None,
            method:         str             = 'robust'
    ) -> "STGCNNormalizer":
        """
        Args:
            y_train (np.ndarray): Training target array.
            train_mask (np.ndarray, optional): Mask to select valid targets.
            method (str): The scaling method to use.
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
    
    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """Normalizes the target array using the fitted target scaler."""
        if self.target_scaler is None:
            raise RuntimeError("Must call fit_target() before transforming.")
        orig_shape = y.shape
        return self.target_scaler.transform(y.reshape(-1, 1)).reshape(orig_shape)
    
    def inverse_transform_target(
            self, 
            predictions: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inverse-transforms predictions back to the original scale. This is crucial for evaluation."""
        if self.target_scaler is None:
            raise RuntimeError("Must call fit_target() before inverse transforming.")
        orig_shape = predictions.shape
        out = self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(orig_shape)
        if isinstance(predictions, torch.Tensor):
            out = torch.as_tensor(out, device=predictions.device, dtype=predictions.dtype)
        return out


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
        self.feature_names:     List[str] = []
        self.feature_scalers:   List[Union[BaseEstimator, None]] = []
        self.feature_log_flags: List[bool] = []

    def fit_features(
            self, 
            x_train:                np.ndarray, 
            feature_names:          List[str], 
            features_to_skip_norm:  List[str]           = None,
            default_method:         str                 = 'robust',
            scaler_map:             Dict[str, str]      = None,
            log_features:           List[str]           = None
    ) -> "Homogeneous":
        """
        Fit scalers to each feature.
        
        Args:
            x_train: np.ndarray of shape (T_train, R, F).
            feature_names: list of feature names (length F).
            features_to_skip_norm: skip normalization if any substring matches.
            default_method: fallback method if no match is found.
            scaler_map: dict {substring -> scaler_method}.
        """
        logger.info("Fitting homogeneous normalizer.")
        self.feature_names = feature_names
        scaler_map = scaler_map or {}
        log_features = log_features or []
        
        for i, feature_name in enumerate(feature_names):
            # Get values
            vals = x_train[..., i].reshape(-1, 1)
            
            # Log rule
            # - NOTE: This can be independent of 'features_to_skip_norm', so comes before.
            apply_log = any(s in feature_name for s in log_features)
            self.feature_log_flags.append(apply_log)
            if apply_log:
                vals = self.apply_log1p_safe(vals=vals, feature_name=feature_name, context="fit")
            
            # Skip rule
            if features_to_skip_norm and any(s in feature_name for s in features_to_skip_norm):
                logger.info(f"Skipping normalization for feature '{feature_name}'")
                self.feature_scalers.append(None)
                continue
                        
            # Scaler map lookup (substring match)
            method = None
            for substr, scaler_name in scaler_map.items():
                if substr in feature_name:
                    method = scaler_name
                    logger.info(f"Feature '{feature_name}': matched on '{substr}' -> '{method}'")
                    break
            if method is None:
                method = default_method
            
            # Get and fit scaler
            scaler = self._get_scaler(method=method)  
            scaler.fit(vals)
            self.feature_scalers.append(scaler)
                        
        logger.info("Finished fitting homogeneous normalizer.")
        return self
    
    def transform_features(self, x: np.ndarray) -> np.ndarray:
        """Applies transformation to the full (T, R, F) numpy array."""
        if not self.feature_scalers:
            raise RuntimeError("Must call fit_features() before transforming.")
        out = x.copy()
        for i, (scaler, do_log) in enumerate(zip(self.feature_scalers, self.feature_log_flags)):
            # Get values
            vals = out[..., i].reshape(-1, 1)
            # Log transform
            if do_log:
                vals = self.apply_log1p_safe(
                    vals=vals,
                    feature_name=self.feature_names[i],
                    context="transform"
                )
            # Apply scaler
            if scaler is not None:
                vals = scaler.transform(vals)
            out[..., i] = vals.reshape(out[..., i].shape)
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
        self.feature_names:     Dict[str, List[str]] = {}
        self.feature_scalers:   Dict[str, List[Union[BaseEstimator, None]]] = {}
        self.feature_log_flags: Dict[str, List[bool]] = {}
    
    def fit_features(
            self,
            x_train:                Dict[int, HeteroData],
            feature_names:          Dict[str, List[str]],
            features_to_skip_norm:  List[str]               = None,
            default_method:         str                     = 'robust',
            scaler_map:             Dict[str, str]          = None,
            log_features:           List[str]               = None,
    ) -> "Heterogeneous":
        """
        Fit scalers per node type and feature.

        Args:
            x_train: Dict[int, HeteroData] snapshots.
            feature_names: {node_type -> [feature_name1, feature_name2, ...]}.
            default_method: scaler method used if not matched.
            features_to_skip_norm: list of substrings; skip if any matches.
            scaler_map: {substring -> scaler_method}.
            log_features: list of substrings; apply log1p if any matches.
        """
        logger.info("Fitting heterogeneous normalizer (no NaN imputation).")
        self.feature_names = feature_names
        scaler_map = scaler_map or {}
        log_features = log_features or []
                
        # 1. Collect node‑wise feature matrices
        collector: Dict[str, List[torch.Tensor]] = {nt: [] for nt in feature_names}
        for snap in x_train.values():
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
            self.feature_log_flags[nt] = []
            
            for i, feature_name in enumerate(names):
                # Get values
                vals = joined[:, i].reshape(-1, 1)
                
                # Log rule
                apply_log = any(s in feature_name for s in log_features)
                self.feature_log_flags[nt].append(apply_log)
                if apply_log:
                    vals = self.apply_log1p_safe(vals=vals, feature_name=feature_name, context="fit")
                
                # Skip rule
                if features_to_skip_norm and any(s in feature_name for s in features_to_skip_norm):
                    logger.info(f"Node '{nt}', feature '{feature_name}': skipped")
                    self.feature_scalers[nt].append(None)
                    continue
                
                # Scaler map lookup (substring match)
                method = None
                for substr, scaler_name in scaler_map.items():
                    if substr in feature_name:
                        method = scaler_name
                        logger.info(f"Node '{nt}', feature '{feature_name}': matched on '{substr}' -> '{method}'")
                        break
                if method is None:
                    method = default_method
                
                scaler = self._get_scaler(method=method)
                scaler.fit(vals)
                self.feature_scalers[nt].append(scaler)
        
        logger.info("Finished fitting heterogeneous normalizer.")
        return self
    
    def transform_features(self, x: Dict[int, HeteroData]) -> Dict[int, HeteroData]:
        """Applies the transformation to all snapshots in-place."""
        if not self.feature_scalers:
            raise RuntimeError("Must call fit_features() before transforming.")
        
        norm_data = {}
        for t, snapshot in x.items():
            norm_snapshot = snapshot.clone()
            for nt in norm_snapshot.node_types:
                if "x" in norm_snapshot[nt]:
                    arr = norm_snapshot[nt].x.cpu().numpy()
                    scalers = self.feature_scalers[nt]
                    log_flags = self.feature_log_flags[nt]
                    
                    for i, (scaler, do_log) in enumerate(zip(scalers, log_flags)):
                        # Get values
                        vals = arr[:, i].reshape(-1, 1)
                        # Log transform
                        if do_log:
                            vals = self.apply_log1p_safe(
                                vals=vals,
                                feature_name=self.feature_names[nt][i],
                                context="transform"
                            )
                        # Apply scaler
                        if scaler is not None:
                            vals = scaler.transform(vals)
                            
                        arr[:, i] = vals.reshape(-1)
                    
                    norm_snapshot[nt].x = torch.tensor(
                        data=arr,
                        device=norm_snapshot[nt].x.device,
                        dtype=norm_snapshot[nt].x.dtype
                    )
            norm_data[t] = norm_snapshot
        return norm_data