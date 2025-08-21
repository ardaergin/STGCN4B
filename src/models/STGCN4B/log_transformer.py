from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData


import logging; logger = logging.getLogger(__name__)


class STGCNLogTransformer(ABC):
    """An abstract base class to log transform data for STGCN models."""
            
    @staticmethod
    def apply_log1p_safe(vals: np.ndarray) -> np.ndarray:
        """Safely apply log1p to an array, clipping negatives to 0."""
        return np.log1p(np.clip(vals, a_min=0, a_max=None))

    @abstractmethod
    def log_transform_features(
            self, 
            x:              Union[np.ndarray, Dict[int, HeteroData]], 
            feature_names:  Union[List[str], Dict[str, List[str]]],
            log_features:   Optional[List[str]] = None
    ):
        """Apply log-transform to features."""
        pass


class Homogeneous(STGCNLogTransformer):
    """
    Log-transformer for homogeneous graph data.
    
    Notes:
    - The input features are a 3D tensor: (T, R, F), F is the feature dimension.
    - Data contains NaNs, zero-imputation is handled later.
    - Data is on CPU, batches are loaded to GPU later in training.
    """
    
    def log_transform_features(
            self, 
            x:              np.ndarray, 
            feature_names:  List[str], 
            log_features:   Optional[List[str]] = None
    ) -> np.ndarray:
        """        
        Apply log-transform to features in homogeneous data.
        
        Args:
            x: np.ndarray of shape (T_train, R, F).
            feature_names: list of feature names (length F).
            log_features: features to log transform (substring or full feature name)
        """
        log_features = log_features or []
        
        out = x.copy()
        
        for i, feature_name in enumerate(feature_names):
            vals = out[..., i].reshape(-1, 1)
            apply_log = any(substr in feature_name for substr in log_features)
            if apply_log:
                vals = self.apply_log1p_safe(vals=vals)
                logger.info(f"Feature '{feature_name}': log1p applied.")
            out[..., i] = vals.reshape(out[..., i].shape)
        logger.info("Finished log-transforming homogeneous features.")
        return out


class Heterogeneous(STGCNLogTransformer):
    """
    Log-transformer for heterogeneous graph data.

    Notes:
    - The input features are HeteroData snapshots: Dict[int, HeteroData].
    - Data contains NaNs, zero-imputation is handled later.
    - Data is on CPU, batches are loaded to GPU later in training.
    """
        
    def log_transform_features(
            self, 
            x:              Dict[int, HeteroData],
            feature_names:  Dict[str, List[str]],
            log_features:   Optional[List[str]] = None
    ) -> Dict[int, HeteroData]:
        """
        Apply log-transform to features in heterogeneous data.

        Args:
            x: Dict[int, HeteroData] snapshots.
            feature_names: {node_type -> [feature_name1, feature_name2, ...]}.
            log_features: list of substrings or names of features to log transform.

        Returns:
            Dict[int, HeteroData]: transformed snapshots.
        """
        log_features = log_features or []

        out = {}
        for t, snapshot in x.items():
            out_snapshot = snapshot.clone()
            for nt in out_snapshot.node_types:
                if "x" not in out_snapshot[nt]:
                    continue
                arr = out_snapshot[nt].x.cpu().numpy()
                names = feature_names.get(nt, [])

                processed_features = []
                for i, feature_name in enumerate(names):
                    vals = arr[:, i].reshape(-1, 1)
                    apply_log = any(substr in feature_name for substr in log_features)
                    if apply_log:
                        vals = self.apply_log1p_safe(vals=vals)
                        logger.info(f"Feature '{feature_name}': log1p applied.")
                    processed_features.append(vals)
                                
                new_arr = np.hstack(processed_features).astype(np.float32)
                out_snapshot[nt].x = torch.tensor(
                    data=new_arr,
                    device=out_snapshot[nt].x.device,
                    dtype=torch.float32
                )

            out[t] = out_snapshot

        logger.info("Finished log-transforming heterogeneous features.")
        return out