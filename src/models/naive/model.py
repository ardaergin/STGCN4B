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
    
    def predict(
            self,
            X: pd.Series,
    ) -> np.ndarray:
        n = len(X)
        h = len(self.forecast_horizons)
        last_vals = X.to_numpy().reshape(n, 1)
        preds = np.tile(last_vals, (1, h))
        return preds
    
    def evaluate(
        self,
        X_test: pd.Series,
        y_test: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Evaluates the persistence model's performance on the test set.
        Ensures evaluation is always done on the final, absolute scale.
        """
        preds = self.predict(X_test) # (n, h)
        targets = y_test.to_numpy()  # (n, h)
                        
        # Calculate per-horizon metrics
        per_horizon_metrics = {}
        for i, horizon in enumerate(self.forecast_horizons):
            # Calculate metrics for the current horizon (indexed by position `i`)
            h_reg_results = regression_results(targets[:, i], preds[:, i])

            # Use the original column name for the dictionary key
            horizon_col_name = str(y_test.columns[i])
            per_horizon_metrics[horizon_col_name] = h_reg_results

            # Log the results for this horizon, matching the main model's style
            logger.info(
                f"Horizon {horizon:>3}: "
                f"MSE={h_reg_results['mse']:.4f} | "
                f"RMSE={h_reg_results['rmse']:.4f} | "
                f"MAE={h_reg_results['mae']:.4f} | "
                f"R²={h_reg_results['r2']:.4f} | "
                f"MAPE={h_reg_results['mape']:.2f}%"
            )

        # Calculate overall metrics on the absolute values
        overall_metrics = regression_results(targets.flatten(), preds.flatten())
        logger.info(f"Overall: "
                    f"MSE={overall_metrics['mse']:.4f} | "
                    f"RMSE={overall_metrics['rmse']:.4f} | "
                    f"MAE={overall_metrics['mae']:.4f} | "
                    f"R²={overall_metrics['r2']:.4f} | "
                    f"MAPE={overall_metrics['mape']:.2f}%")

        # Assemble final dictionaries
        metrics = {**overall_metrics, "per_horizon_metrics": per_horizon_metrics}
        model_outputs = {
            "predictions": preds,
            "targets": targets,
        }

        return metrics, model_outputs