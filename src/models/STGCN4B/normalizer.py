from abc import ABC, abstractmethod
from typing import Union, List, Dict
import numpy as np
import torch
from torch_geometric.data import HeteroData

import logging; logger = logging.getLogger(__name__)


class STGCNNormalizer(ABC):
    """
    An abstract base class for data normalization in STGCN models.
    
    It defines a common interface for feature normalization and provides a concrete
    implementation for target normalization, which is shared across different
    graph types.
    """
    def __init__(self):
        # Targets (implemented in base class)
        self.target_center = None
        self.target_scale = None

        # Features (implemented in subclasses)
        self.feature_center = None
        self.feature_scale = None
    
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
            train_targets: np.ndarray, 
            train_mask: np.ndarray = None,
            method: str = 'median'
    ) -> "STGCNNormalizer":
        """
        Calculates target statistics from the training data using the specified method.
        
        Args:
            train_targets (np.ndarray): Training target array.
            train_mask (np.ndarray, optional): Mask to select valid targets.
            method (str): The scaling method to use: 'mean' or 'median'.
        """
        if train_mask is not None:
            # Ensure mask is boolean for indexing
            valid_targets = train_targets[train_mask.astype(bool)]
        else:
            valid_targets = train_targets
        
        if valid_targets.size == 0:
            self.target_center, self.target_scale = 0.0, 1.0
            logger.warning("No valid targets found for fitting the scaler. Using default values (0, 1).")
            return self

        if method == 'mean':
            self.target_center = np.mean(valid_targets)
            self.target_scale = np.std(valid_targets)
            log_method = "Mean"
        elif method == 'median':
            self.target_center = np.median(valid_targets)
            q25 = np.percentile(valid_targets, 25)
            q75 = np.percentile(valid_targets, 75)
            self.target_scale = q75 - q25
            log_method = "Median/IQR"
        else:
            raise ValueError(f"Unknown scaling method for target: {method}. Choose 'mean' or 'median'.")
        
        # Avoid division by zero if target is constant
        if self.target_scale == 0:
            self.target_scale = 1.0
            
        logger.info(f"Fitted target processor using {log_method} scaling. Center: {self.target_center:.4f}, Scale: {self.target_scale:.4f}")
        return self

    def transform_target(
            self, 
            targets: np.ndarray
    ) -> np.ndarray:
        """Normalizes the target array using the fitted target scaler."""
        if self.target_center is None:
            raise RuntimeError("Must call fit_target() before transforming.")
        return (targets - self.target_center) / self.target_scale
    
    def inverse_transform_target(
            self, 
            predictions: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inverse-transforms predictions back to the original scale. This is crucial for evaluation."""
        if self.target_center is None:
            raise RuntimeError("Must call fit_target() before inverse transforming.")
        return (predictions * self.target_scale) + self.target_center



class Homogeneous(STGCNNormalizer):
    """
    Normalizer for homogeneous graph data represented by a single feature tensor.
    
    Note that the Homogeneous normalizer does not affect the underlying data 
    because basic arithmetic operations in NumPy create new arrays by default.
    """
    
    def __init__(self):
        super().__init__()
        self.feature_center = None
        self.feature_scale = None
        
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
        if method == 'mean':
            self.feature_center = np.nanmean(train_data, axis=(0, 1))
            self.feature_scale = np.nanstd(train_data, axis=(0, 1))
            logger.info("Using mean-based (Z-score) scaling.")
        elif method == 'median':
            self.feature_center = np.nanmedian(train_data, axis=(0, 1))
            q25 = np.nanpercentile(train_data, 25, axis=(0, 1))
            q75 = np.nanpercentile(train_data, 75, axis=(0, 1))
            self.feature_scale = q75 - q25
            logger.info("Using median-based (Robust) scaling.")
        else:
            raise ValueError(f"Unknown scaling method: {method}. Choose 'mean' or 'median'.")
        
        # Identify and handle features to skip (this logic is the same)
        if features_to_skip_norm:
            skip_indices = [i for i, name in enumerate(feature_names) if any(s in name for s in features_to_skip_norm)]
            if skip_indices:
                skipped_feature_names = [feature_names[i] for i in skip_indices]
                logger.info(f"Skipping normalization for {len(skip_indices)} features: {skipped_feature_names}")
                self.feature_center[skip_indices] = 0.0
                self.feature_scale[skip_indices] = 1.0
        
        # Avoid division by zero for constant features
        self.feature_scale[self.feature_scale == 0] = 1.0
        
        logger.info("Fitted feature processor.")
        return self
    
    def transform_features(
            self, 
            all_data: np.ndarray
    ) -> np.ndarray:
        """Applies transformation to the full (T, R, F) numpy array."""
        if self.feature_center is None:
            raise RuntimeError("Must call fit_features() before transforming.")
        return (all_data - self.feature_center) / self.feature_scale



class Heterogeneous(STGCNNormalizer):
    """Normalizer for heterogeneous graph data stored in HeteroData snapshots."""
    
    def __init__(self):
        super().__init__()
        self.feature_center: Dict[str, torch.Tensor] = {}
        self.feature_scale: Dict[str, torch.Tensor] = {}
    
    @staticmethod
    def _nanquantile_torch(x: torch.Tensor, q: float) -> torch.Tensor:
        """
        torch.nanquantile is available from PyTorch 2.2.  If the user sits on an
        older version we emulate it (feature‑wise) via masking.
        """
        if hasattr(torch, "nanquantile"):
            return torch.nanquantile(x, q, dim=0)
        valid = ~torch.isnan(x)
        # need at least one valid value per feature
        x_safe = torch.where(valid, x, torch.tensor(float("nan"), device=x.device))
        return torch.tensor(
            [torch.nanquantile(col, q).item() for col in x_safe.T], device=x.device
        )
    
    def fit_features(
            self,
            train_data: Dict[int, HeteroData],
            feature_names: Dict[str, List[str]],
            method: str = "median",
            features_to_skip_norm: List[str] | None = None,
    ) -> "Heterogeneous":
        logger.info("Fitting heterogeneous normalizer (no NaN imputation).")

        # 1. collect node‑wise feature matrices
        collector: Dict[str, List[torch.Tensor]] = {nt: [] for nt in feature_names}
        for snap in train_data.values():
            for nt in snap.node_types:
                if "x" in snap[nt]:
                    collector[nt].append(snap[nt].x.float())

        # 2. compute stats
        for nt, tensors in collector.items():
            if not tensors:
                continue
            joined = torch.cat(tensors, 0)  # (N_total, C)

            # statistics ignoring NaNs
            if method == "mean":
                self.feature_center[nt] = torch.nanmean(joined, dim=0)
                self.feature_scale[nt] = torch.nanstd(joined,  dim=0)
            elif method == "median":
                self.feature_center[nt] = torch.nanmedian(joined, dim=0).values
                q25 = self._nanquantile_torch(joined, 0.25)
                q75 = self._nanquantile_torch(joined, 0.75)
                self.feature_scale[nt] = q75 - q25
            else:
                raise ValueError("method must be 'mean' or 'median'")

            # 3. skip‑normalisation handling
            if features_to_skip_norm:
                names = feature_names.get(nt, [])
                skip_indices = [i for i, n in enumerate(names)
                        if any(s in n for s in features_to_skip_norm)]
                if skip_indices:
                    skipped_feature_names = [names[i] for i in skip_indices]
                    logger.info(f"Node type '{nt}': Skipping normalization for {len(skip_indices)} features: {skipped_feature_names}")
                    self.feature_center[nt][skip_indices] = 0.0
                    self.feature_scale[nt][skip_indices] = 1.0

            # 4. avoid divide‑by‑zero
            self.feature_scale[nt][self.feature_scale[nt] == 0] = 1.0

        return self
    
    def transform_features(
            self, 
            all_data: Dict[int, HeteroData]
    ) -> Dict[int, HeteroData]:
        """Applies the transformation to all snapshots in-place."""
        if not self.feature_center:
            raise RuntimeError("Must call fit_features() before transforming.")
        
        norm_data = {}
        for t, snapshot in all_data.items():
            norm_snapshot = snapshot.clone()
            for nt in norm_snapshot.node_types:
                if 'x' in norm_snapshot[nt]:
                    norm_snapshot[nt].x = (
                        norm_snapshot[nt].x - self.feature_center[nt]
                        ) / self.feature_scale[nt]
            norm_data[t] = norm_snapshot
        return norm_data