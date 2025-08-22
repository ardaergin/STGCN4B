import os
from abc import ABC, abstractmethod
from typing import Union, List, Dict
from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer
)
import matplotlib.pyplot as plt
import seaborn as sns

import logging; logger = logging.getLogger(__name__)


class STGCNNormalizer(ABC):
    """
    An abstract base class for data normalization in STGCN models.
    
    It defines a common interface for feature normalization and provides a concrete
    implementation for target normalization, which is shared across different
    graph types.
    """
    def __init__(self, plot_dist: bool = False, plot_dir: str = None):
        self.target_scaler = None

        # Plotting
        self.plot_dist = plot_dist
        self.plot_dir = plot_dir
        if self.plot_dir:
            os.makedirs(self.plot_dir, exist_ok=True)
    
    @staticmethod
    def _get_scaler(method: str):
        """
        Factory function that returns a new scaler instance based on the method.
                
        Returns:
            An un-fitted sklearn scaler instance.
        """
        if method == "none":
            return None # do nothing
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
    
    @abstractmethod
    def fit_features(self):
        """Abstract method to calculate feature statistics from training data."""
        pass
    
    @abstractmethod
    def transform_features(self):
        """Abstract method to apply feature transformation."""
        pass
    
    def _get_feature_stats(self):
        """Logging of statistics for each feature."""
        pass
    
    def fit_target(
            self, 
            y_train:        np.ndarray, 
            y_train_mask:   np.ndarray      = None,
            method:         str             = 'robust'
    ) -> "STGCNNormalizer":
        """
        Args:
            y_train: Training target array.
            y_train_mask: Mask to select valid targets.
            method: The scaling method to use.
        """
        if method == "none":
            self.target_scaler = None
            logger.info("Target normalization disabled (method=none).")
            return self
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
            return y
        orig_shape = y.shape
        return self.target_scaler.transform(y.reshape(-1, 1)).reshape(orig_shape)
    
    def inverse_transform_target(
            self, 
            predictions: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inverse-transforms predictions back to the original scale."""
        if self.target_scaler is None:
            return predictions
        orig_shape = predictions.shape
        out = self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(orig_shape)
        if isinstance(predictions, torch.Tensor):
            out = torch.as_tensor(out, device=predictions.device, dtype=predictions.dtype)
        return out
    
    def _get_target_stats(self, y: np.ndarray):
        """Logs statistics for the normalized target array."""
        logger.info(f"Normalized target shape: {y.shape}")
        logger.info("Normalized target stats:")
        
        # (T, H) or (T, N, H) depending on task
        if y.ndim == 2:  # (time, horizon)
            for i in range(y.shape[1]): # per horizon
                vals = y[:, i].ravel()
                min_val, max_val, mean_val, std_val = np.nanmin(vals), np.nanmax(vals), np.nanmean(vals), np.nanstd(vals)
                self._log_stats(f"Target (Horizon {i:<2}): ", mean_val, std_val, min_val, max_val)
                if self.plot_dist:
                    self._plot_distribution(vals, f"Target Distribution Horizon {i}", self.plot_dir, f"target_h{i}")
        
        elif y.ndim == 3:  # (time, nodes, horizon)
            for i in range(y.shape[2]): # per horizon
                vals = y[:, :, i].ravel()
                min_val, max_val, mean_val, std_val = np.nanmin(vals), np.nanmax(vals), np.nanmean(vals), np.nanstd(vals)
                self._log_stats(f"Target (Horizon {i:<2}): ", mean_val, std_val, min_val, max_val)
                if self.plot_dist:
                    self._plot_distribution(vals, f"Target Distribution Horizon {i}", self.plot_dir, f"target_h{i}")
        else:
            logger.warning(f"Unexpected target shape {y.shape}, skipping detailed stats.")
    
    @staticmethod
    def _log_stats(name: str, mean_val: float, std_val: float, min_val: float, max_val: float):
        """Pretty logs per-feature statistics with grouped pairs (min/max, mean/std)."""
        logger.info(
            f"    {name:<25} | "
            f"min/max: {min_val:>8.4f} / {max_val:<8.4f} | "
            f"mean/std: {mean_val:>8.4f} / {std_val:<8.4f}"
        )
    
    @staticmethod
    def _plot_distribution(
        data: np.ndarray,
        title: str,
        dir: str,
        plot_name: str
    ) -> None:
        """
        Helper method to create and save a distribution plot.
        
        Args:
            data (np.ndarray): The 1D array of data to plot.
            title (str): The title for the plot.
            dir (str): The directory to save the plot image.
            plot_name (str): The name of the plot image file (without extension).
        """
        
        if data.size == 0:
            logger.warning(f"No valid data provided for plotting '{title}'. Skipping.")
            return
        
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data, kde=True, bins=50)
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plot_path = os.path.join(dir, plot_name + ".png")
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to create plot '{title}': {e}")



class Homogeneous(STGCNNormalizer):
    """
    Normalizer for homogeneous graph data.
    
    Notes:
    - The input features are a 3D tensor: (T, R, F), F is the feature dimension.
    - Data contains NaNs, zero-imputation is handled later.
    - Data is on CPU, batches are loaded to GPU later in training.
    """
    
    def __init__(self, plot_dist: bool = False, plot_dir: str = "plots"):
        super().__init__(plot_dist=plot_dist, plot_dir=plot_dir)
        self.feature_names:     List[str] = []
        self.feature_scalers:   List[Union[BaseEstimator, None]] = []

    def fit_features(
            self, 
            x_train:                np.ndarray, 
            feature_names:          List[str], 
            features_to_skip_norm:  List[str]           = None,
            default_method:         str                 = 'robust',
            scaler_map:             Dict[str, str]      = None,
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
        
        for i, feature_name in enumerate(feature_names):
            # Get values
            vals = x_train[..., i].reshape(-1, 1)
                        
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
        for i, scaler in enumerate(self.feature_scalers):
            if scaler is not None:
                col_data = out[..., i].reshape(-1, 1)
                out[..., i] = scaler.transform(col_data).reshape(out[..., i].shape)
        return out
    
    def _get_feature_stats(
            self, 
            x:              np.ndarray,
            feature_names:  List[str], 
    ) -> None:
        """Logs per-feature statistics for the normalized homogeneous feature array."""
        logger.info(f"Normalized features shape: {x.shape}")  
        logger.info("Normalized homogeneous features stats (per-feature):")
        
        num_features = x.shape[2]
        for i in range(num_features):
            name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
            vals = np.asarray(x[:, :, i])
            min_val, max_val, mean_val, std_val = np.nanmin(vals), np.nanmax(vals), np.nanmean(vals), np.nanstd(vals)
            self._log_stats(name, mean_val, std_val, min_val, max_val)
            if self.plot_dist:
                self._plot_distribution(vals, f"Feature Distribution {name}", self.plot_dir, f"feature_{i}_{name}")



class Heterogeneous(STGCNNormalizer):
    """
    Normalizer for heterogeneous graph data.

    Notes:
    - The input features are HeteroData snapshots: Dict[int, HeteroData].
    - Data contains NaNs, zero-imputation is handled later.
    - Data is on CPU, batches are loaded to GPU later in training.
    """
    
    def __init__(self, plot_dist: bool = False, plot_dir: str = "plots"):
        super().__init__(plot_dist=plot_dist, plot_dir=plot_dir)
        self.feature_names:     Dict[str, List[str]] = {}
        self.feature_scalers:   Dict[str, List[Union[BaseEstimator, None]]] = {}
    
    def fit_features(
            self,
            x_train:                Dict[int, HeteroData],
            feature_names:          Dict[str, List[str]],
            features_to_skip_norm:  List[str]               = None,
            default_method:         str                     = 'robust',
            scaler_map:             Dict[str, str]          = None,
    ) -> "Heterogeneous":
        """
        Fit scalers per node type and feature.

        Args:
            x_train: Dict[int, HeteroData] snapshots.
            feature_names: {node_type -> [feature_name1, feature_name2, ...]}.
            default_method: scaler method used if not matched.
            features_to_skip_norm: list of substrings; skip if any matches.
            scaler_map: {substring -> scaler_method}.
        """
        logger.info("Fitting heterogeneous normalizer (no NaN imputation).")
        self.feature_names = feature_names
        scaler_map = scaler_map or {}
                        
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
                        
            for i, feature_name in enumerate(names):
                # Get values
                vals = joined[:, i].reshape(-1, 1)
                
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
                if "x" in norm_snapshot[nt] and nt in self.feature_scalers:
                    original_tensor = norm_snapshot[nt].x
                    arr = original_tensor.cpu().numpy()
                    scalers = self.feature_scalers[nt]
                         
                    # Process each column and store it in a list to avoid dtype issues
                    processed_cols = []
                    for i, scaler in enumerate(scalers):
                        vals = arr[:, i].reshape(-1, 1)
                        if scaler is not None:
                            vals = scaler.transform(vals)
                        
                        processed_cols.append(vals)
                    
                    # Combine processed columns into a new array and create a tensor
                    new_arr = np.hstack(processed_cols).astype(np.float32)
                    norm_snapshot[nt].x = torch.tensor(
                        data=new_arr,
                        device=original_tensor.device,
                        dtype=torch.float32
                    )
            
            norm_data[t] = norm_snapshot
        return norm_data
    
    def _get_feature_stats(
            self, 
            x:              Dict[int, HeteroData], 
            feature_names:  Dict[str, List[str]],
    ) -> None:
        """Logs per-feature statistics for the normalized heterogeneous graph snapshots."""
        logger.info("Normalized feature shapes (sample from first 2 snapshots):")
        for t, snap in list(x.items())[:2]:
            for nt in snap.node_types:
                if "x" in snap[nt] and snap[nt].x is not None:
                    logger.info(f"  [t={t}] Node type '{nt}': {snap[nt].x.shape}")
        
        logger.info("Normalized heterogeneous features stats (per-feature):")
        tensors_by_nodetype = defaultdict(list)
        for snapshot in x.values():
            for node_type in snapshot.node_types:
                if 'x' in snapshot[node_type] and snapshot[node_type].x is not None:
                    tensors_by_nodetype[node_type].append(snapshot[node_type].x)
        
        for node_type, tensor_list in tensors_by_nodetype.items():
            if not tensor_list:
                logger.info(f"  - Node Type '{node_type}': No feature tensors found.")
                continue
            
            combined_tensor = torch.cat(tensor_list, dim=0)
            feature_names_list = feature_names.get(node_type, [])
            num_features = combined_tensor.shape[1]
            
            logger.info(f"  --- Stats for Node Type: '{node_type}' ({num_features} features) ---")
            
            for i in range(num_features):
                name = feature_names_list[i] if i < len(feature_names_list) else f"Feature_{i}"
                feature_col = combined_tensor[:, i]
                valid_values = feature_col[~torch.isnan(feature_col)]
                
                if valid_values.numel() > 0:
                    vals = valid_values.cpu().numpy()
                    min_val, max_val, mean_val, std_val = np.min(vals), np.max(vals), np.mean(vals), np.std(vals)
                    self._log_stats(name, mean_val, std_val, min_val, max_val)
                    if self.plot_dist:
                        self._plot_distribution(vals, f"{node_type} Feature Distribution {name}", self.plot_dir, f"{node_type}_feature_{i}_{name}")
                else:
                    logger.info(f"    Feature '{name:<20}': All NaN values.")
