from typing import Union
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
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None

    def fit_features(self, train_array: np.ndarray):
        """
        Calculates feature statistics (mean, std) from the training data slice.
        
        Args:
            train_array (np.ndarray): A 3D array of shape (T_train, R, F)
                                      containing the training features.
        """
        # Calculate mean and std along the time and room axes (0, 1), ignoring NaNs.
        # This results in a 1D array of shape (F,) for each statistic.
        self.feature_mean = np.nanmean(train_array, axis=(0, 1))
        self.feature_std = np.nanstd(train_array, axis=(0, 1))

        # Avoid division by zero for constant features
        self.feature_std[self.feature_std == 0] = 1.0
        logger.info("Fitted feature processor.")
        return self

    def transform_features(self, full_array: np.ndarray) -> np.ndarray:
        """
        Applies z-score normalization to the full feature array using the stored stats.
        
        Args:
            full_array (np.ndarray): The complete 3D feature array (T, R, F).
        
        Returns:
            np.ndarray: The normalized feature array.
        """
        if self.feature_mean is None:
            raise RuntimeError("Must call fit_features() before transforming.")
        return (full_array - self.feature_mean) / self.feature_std

    def fit_target(self, train_targets: np.ndarray, train_mask: np.ndarray = None):
        """
        Calculates target statistics from the training data.
        
        Args:
            train_targets (np.ndarray): Training target array.
            train_mask (np.ndarray, optional): Mask to select valid targets.
        """
        if train_mask is not None:
            valid_targets = train_targets[train_mask.astype(bool)]
        else:
            valid_targets = train_targets
        
        if valid_targets.size == 0:
            self.target_mean, self.target_std = 0.0, 1.0
        else:
            self.target_mean = np.mean(valid_targets)
            self.target_std = np.std(valid_targets)
        
        if self.target_std == 0:
            self.target_std = 1.0
        logger.info(f"Fitted target processor. Mean: {self.target_mean:.4f}, Std: {self.target_std:.4f}")
        return self

    def transform_target(self, targets: np.ndarray) -> np.ndarray:
        """Normalizes the target array."""
        return (targets - self.target_mean) / self.target_std
    
    def inverse_transform_target(self, predictions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Inverse-transforms predictions back to the original scale."""
        return (predictions * self.target_std) + self.target_mean