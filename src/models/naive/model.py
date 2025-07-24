import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

from ...utils.metrics import regression_results

import logging; logger = logging.getLogger(__name__)


class NaivePersistenceModel:
    """
    A pure persistence baseline for regression forecasts.

    We do not need any delta reconstruction logic here, as the persistence model
    simply predicts the last observed value for each horizon.
    """
    def evaluate_model(
        self,
        X_test: pd.Series,
        y_test: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Evaluates the persistence model's performance on the test set.
        Ensures evaluation is always done on the final, absolute scale.

        Note: 
        We don't really need to reconstruct the predictions here. Since our prediction 
        is delta=0, so our prediction is equal to the source.
        """
        source          = X_test.to_numpy()
        preds_abs       = source # just to be explicit
        targets_delta   = y_test.to_numpy()
        targets_abs     = preds_abs + targets_delta
        
        # Calculate metrics
        metrics = regression_results(y_true=targets_abs, y_pred=preds_abs)

        # Log the final metrics
        logger.info(
            f"Persistence Metrics: "
            f"MSE={metrics['mse']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"MAE={metrics['mae']:.4f} | "
            f"R²={metrics['r2']:.4f} | "
            f"MAPE={metrics['mape']:.2f}%"
        )
        
        # Assemble model outputs
        model_outputs = {
            "predictions": preds_abs,
            "targets": targets_abs,
        }

        return metrics, model_outputs