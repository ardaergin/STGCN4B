import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import joblib

# Logging setup
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TabularDataset:
    """
    Helper class to encapsulate a tabular DataFrame plus target and 
    train/val/test indices, providing easy access to splits.
    """
    def __init__(self,
                 features_df: pd.DataFrame,
                 consumption_df: pd.DataFrame,
                 workhour_labels: np.ndarray,
                 train_idx: List[int],
                 val_idx: List[int],
                 test_idx: List[int],
                 blocks: Dict[int, List]):
        
        # X
        self.feature_df = features_df

        # y
        self.consumption_df = consumption_df
        self.workhour_labels = workhour_labels
        
        # Splits
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.blocks = blocks

    def save(self, path: str = "data/processed/tabular_dataset.joblib") -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = "data/processed/tabular_dataset.joblib") -> "TabularDataset":
        return joblib.load(path)

    def set_mode(self, mode: str):
        if mode == "forecasting":
            self.activate_forecasting_mode()
        elif mode == "classification":
            self.activate_classification_mode()
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def activate_classification_mode(self):
        """
        Set the dataset into classification mode.

        Note:
            - Any column including "consumption" is dropped: 
              i.e., the target column, as well as its MA and lag columns derived from it.
        """
        self.task = "classification"

        # y
        self.y = self.workhour_labels.astype(np.float32)
        self.y_train = self.y[self.train_idx]
        self.y_val = self.y[self.val_idx]
        self.y_test = self.y[self.test_idx]

        # Excluding the consumption-related columns & time-related columns (and "bucket_idx" as usual)
        ## Reason:
        ## - consumption is not a part of the original evaluation in the OfficeGraph paper
        ## - time features are direct leakage for the workhour classification
        cols = self.feature_df.columns
        consumption_columns = [col for col in cols if "consumption" in col]
        time_columns = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        exclude_cols = consumption_columns + time_columns + ['bucket_idx']
        feature_cols = [col for col in cols if col not in exclude_cols]
        self.X = self.feature_df[feature_cols].astype(np.float32).reset_index(drop=True)

        # X
        self.X_train = self.X.iloc[self.train_idx].reset_index(drop=True)
        self.X_val = self.X.iloc[self.val_idx].reset_index(drop=True)
        self.X_test = self.X.iloc[self.test_idx].reset_index(drop=True)
        
    def activate_forecasting_mode(self, 
                                  horizon: int = 1, 
                                  ma_window: Optional[int] = None):
        """
        Set the dataset into forecasting mode.
        Option to do multi-horizon forecasting of consumption
        (e.g., forecasting the average consumption in the next 4 time buckets)

        Also, first computing the full target array and feature matrix,
        then dropping any rows (and their feature‐rows) where the target is NaN.
        This is needed since we did shifting for each block when calculating the targets.
        
        Args:
            horizon: how many buckets ahead (default=1)
            ma_window: if None → 1‐step ahead; if int → average over next `ma_window` buckets
        """
        self.task = "forecasting"

        # 1) Compute full target and full feature‐matrix
        full_y = self.create_target_consumption_values(
            horizon=horizon, 
            ma_window=ma_window
            ).astype(np.float32)
        full_X = (
                self.feature_df
                .drop(columns=["bucket_idx"])
                .astype(np.float32)
                .reset_index(drop=True)
        )

        # 2) Build unfiltered splits
        y_train_all = full_y[self.train_idx]
        y_val_all   = full_y[self.val_idx]
        y_test_all  = full_y[self.test_idx]

        X_train_all = full_X.iloc[self.train_idx].reset_index(drop=True)
        X_val_all   = full_X.iloc[self.val_idx].reset_index(drop=True)
        X_test_all  = full_X.iloc[self.test_idx].reset_index(drop=True)

        # 3) Filter out any NaNs in each split
        train_mask = ~np.isnan(y_train_all)
        val_mask   = ~np.isnan(y_val_all)
        test_mask  = ~np.isnan(y_test_all)

        self.y_train = y_train_all[train_mask]
        self.X_train = X_train_all[train_mask]

        self.y_val   = y_val_all[val_mask]
        self.X_val   = X_val_all[val_mask]

        self.y_test  = y_test_all[test_mask]
        self.X_test  = X_test_all[test_mask]

        # 4) Also store the filtered full arrays
        self.y = full_y[~np.isnan(full_y)]
        self.X = full_X[~np.isnan(full_y)].reset_index(drop=True)

    def create_target_consumption_values(self,
                                        horizon: int = 1,
                                        ma_window: Optional[int] = None
                                        ) -> np.ndarray:
        """
        Build the forecasting target values as a numpy array, per block, to avoid leakage.
                
        Args:
            horizon: how many buckets ahead (default=1)
            ma_window: if None → 1-step ahead; if int → average over next `ma_window` buckets
        
        Returns:
            np.ndarray.
        """        
        df = self.consumption_df.copy()

        if ma_window is None:
            # Simple next‐step target
            target_consumption_values = (
                df
                .groupby("block_id")["consumption"]
                .shift(-horizon)
            ).values
        else:
            # Forward‐looking average over the next ma_window buckets
            def _fwd_ma(s):
                return s.shift(-horizon).rolling(window=ma_window, min_periods=ma_window).mean().shift(ma_window - 1)
            
            target_consumption_values = (
                df
                .groupby("block_id")["consumption"]
                .apply(_fwd_ma)
                .reset_index(level=0, drop=True)
            ).values
        
        return target_consumption_values
    
    ############################################################################################
    # NOTE: I did not incorporate the functions below into the 'activate_forecasting_mode' yet.
    ############################################################################################

    def create_multi_horizon_targets(self,
                                    horizons: List[int] = None,
                                    ma_windows: List[int] = None
                                    ) -> Dict[str, np.ndarray]:
        """
        Create multiple forecasting targets for different horizons.
        Useful for multi-step ahead forecasting models.
        """
        # Defaults:
        if horizons is None: horizons = [1,6,12,24]
        if ma_windows is None: ma_windows = [1,3,6]

        df = self.consumption_df.copy()
        
        targets_dict = {}
        
        for horizon in horizons:
            for ma_window in ma_windows:
                if ma_window == 1:
                    # Single-step target
                    target_name = f"consumption_h{horizon}"
                    target_values = (
                        df.groupby("block_id")["consumption"]
                        .shift(-horizon)
                    ).values
                else:
                    # Multi-step average target
                    target_name = f"consumption_h{horizon}_ma{ma_window}"
                    def _fwd_ma(s):
                        return (s.shift(-horizon)
                            .rolling(window=ma_window, min_periods=ma_window)
                            .mean()
                            .shift(ma_window - 1))
                    
                    target_values = (
                        df.groupby("block_id")["consumption"]
                        .apply(_fwd_ma)
                        .reset_index(level=0, drop=True)
                    ).values
                
                targets_dict[target_name] = target_values
        
        return targets_dict

    def create_classification_targets(self,
                                    thresholds: Optional[List[float]] = None,
                                    quantile_based: bool = True
                                    ) -> Dict[str, np.ndarray]:
        """
        Create classification targets for high/low consumption prediction.
        """
        df = self.consumption_df.copy()
        
        if thresholds is None:
            if quantile_based:
                # Use quantiles from training data
                train_data = df.iloc[self.train_idx]["consumption"]
                thresholds = [
                    train_data.quantile(0.33),  # Low
                    train_data.quantile(0.67)   # High
                ]
            else:
                thresholds = [df["consumption"].mean()]
        
        classification_targets = {}
        
        # Binary classification (above/below threshold)
        for i, threshold in enumerate(thresholds):
            target_name = f"consumption_binary_t{i}"
            classification_targets[target_name] = (df["consumption"] > threshold).astype(int).values
        
        # Multi-class classification
        if len(thresholds) > 1:
            target_name = "consumption_multiclass"
            targets = np.zeros(len(df))
            for i, threshold in enumerate(sorted(thresholds)):
                targets[df["consumption"] > threshold] = i + 1
            classification_targets[target_name] = targets
        
        return classification_targets

    def create_directional_targets(self,
                                horizons: List[int] = None
                                ) -> Dict[str, np.ndarray]:
        """
        Create directional prediction targets (up/down/stable).
        """
        if horizons is None: horizons = [1,6,12,24]

        df = self.consumption_df.copy()
        
        directional_targets = {}
        
        for horizon in horizons:
            # Calculate future change
            future_values = df.groupby("block_id")["consumption"].shift(-horizon)
            current_values = df["consumption"]
            
            pct_change = (future_values - current_values) / (current_values + 1e-8)
            
            # Create directional labels
            # 0: Down (< -threshold), 1: Stable (within threshold), 2: Up (> threshold)
            threshold = 0.05  # 5% change threshold
            directional = np.ones(len(df))  # Default to stable
            directional[pct_change < -threshold] = 0  # Down
            directional[pct_change > threshold] = 2   # Up
            
            target_name = f"consumption_direction_h{horizon}"
            directional_targets[target_name] = directional
        
        return directional_targets

    def exclude_feature_columns(self, exclude_cols: List[str]):
        self.X = self.X.drop(exclude_cols)

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Return a quick DataFrame summarizing each feature's dtype,
        % missing, and basic stats (mean/std).
        """
        stats = pd.DataFrame({
            'dtype': self.X.dtypes,
            'pct_missing': self.X.isna().mean(),
            'mean': self.X.mean(numeric_only=True),
            'std': self.X.std(numeric_only=True)
        })
        return stats