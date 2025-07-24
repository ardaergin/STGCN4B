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
    def __init__(
            self, 
            args: Any,
    ):
        self.forecast_horizons = args.forecast_horizons
        
    def evaluate_model(
        self,
        X_test: pd.Series,
        y_test: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Evaluates the persistence model's performance on the test set.
        Ensures evaluation is always done on the final, absolute scale.
        """
        preds =   X_test.to_numpy()
        targets = y_test.to_numpy()
        metrics = regression_results(targets, preds)
        
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
            "predictions": preds,
            "targets": targets,
        }

        return metrics, model_outputs