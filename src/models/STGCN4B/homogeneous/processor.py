from typing import Union, List
import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)

class NumpyDataProcessor:
    """
    A simple and fast data processor that works exclusively with NumPy arrays.
    It calculates statistics on a training slice while ignoring NaNs, and then
    applies the transformation to a full array in a vectorized manner.
    """
    def __init__(self):
        self.feature_center = None
        self.feature_scale = None
        self.target_center = None
        self.target_scale = None

    def fit_features(self, 
                     train_array: np.ndarray, 
                     feature_names: List[str], 
                     method: str = 'median',
                     features_to_skip_norm: List[str] = None):
        """
        Calculates feature statistics from the training data slice.

        Args:
            train_array (np.ndarray): A 3D array of shape (T_train, R, F).
            feature_names (List[str]): A list of length F with feature names.
            method (str): The scaling method to use: 'mean' for Z-score or 'median' for Robust Scaling.
            features_to_skip_norm (List[str], optional): List of substrings for features to skip.
        """
        if method == 'mean':
            self.feature_center = np.nanmean(train_array, axis=(0, 1))
            self.feature_scale = np.nanstd(train_array, axis=(0, 1))
            logger.info("Using mean-based (Z-score) scaling.")
        elif method == 'median':
            self.feature_center = np.nanmedian(train_array, axis=(0, 1))
            q25 = np.nanpercentile(train_array, 25, axis=(0, 1))
            q75 = np.nanpercentile(train_array, 75, axis=(0, 1))
            self.feature_scale = q75 - q25
            logger.info("Using median-based (Robust) scaling.")
        else:
            raise ValueError(f"Unknown scaling method: {method}. Choose 'mean' or 'median'.")
        
        # Identify and handle features to skip (this logic is the same)
        if features_to_skip_norm:
            skip_indices = [i for i, name in enumerate(feature_names) if any(s in name for s in features_to_skip_norm)]
            if skip_indices:
                self.feature_center[skip_indices] = 0.0
                self.feature_scale[skip_indices] = 1.0
                logger.info(f"Skipping normalization for {len(skip_indices)} features.")

        # Avoid division by zero for constant features
        self.feature_scale[self.feature_scale == 0] = 1.0
        
        logger.info("Fitted feature processor.")
        return self

    def transform_features(self, full_array: np.ndarray) -> np.ndarray:
        """
        Applies the fitted transformation to the full feature array using the stored stats.
        
        Args:
            full_array (np.ndarray): The complete 3D feature array (T, R, F).
        
        Returns:
            np.ndarray: The normalized feature array.
        """
        if self.feature_center is None:
            raise RuntimeError("Must call fit_features() before transforming.")
        return (full_array - self.feature_center) / self.feature_scale

    def fit_target(self, 
                   train_targets: np.ndarray, 
                   train_mask: np.ndarray = None,
                   method: str = 'median'):
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

    def transform_target(self, targets: np.ndarray) -> np.ndarray:
        """Normalizes the target array using the fitted target scaler."""
        if self.target_center is None:
            raise RuntimeError("Must call fit_target() before transforming.")
        return (targets - self.target_center) / self.target_scale
    
    def inverse_transform_target(self, predictions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse-transforms predictions back to the original scale. This is crucial for evaluation.
        """
        if self.target_center is None:
            raise RuntimeError("Must call fit_target() before inverse transforming.")
        return (predictions * self.target_scale) + self.target_center
