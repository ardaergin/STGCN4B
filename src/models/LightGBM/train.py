import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .model import LGBMWrapper
from ...utils.tracking import TrainingHistory, TrainingResult
from ...utils.metrics import (regression_results, binary_classification_results, find_optimal_f1_threshold)

import logging; logger = logging.getLogger(__name__)


class LGBMTrainer:
    """
    A trainer class to handle the training and evaluation of the LGBMWrapper.
    """
    def __init__(self, args: Any):
        """
        Args:
            args: The experiment configuration object, also includes the model parameters for LGBM.
        """
        self.args = args

        # Getting valid model parameters from args
        valid_params = LGBMWrapper().get_params().keys()
        self.model_params = {
            k: v for k, v in vars(self.args).items() if k in valid_params
        }
        
        # Set metric & objective
        if self.args.task_type == "workhour_classification":
            self.model_params.update(
                objective = "binary",
                metric    = ["auc", "binary_logloss"],
                is_unbalance = True,
            )
        else: # Forecasting tasks
            if args.forecast_loss_func == "mae":
                self.model_params.update(
                    objective = "regression_l1", # mae
                    metric    = "l1",
                )
            elif args.forecast_loss_func == "mse":
                self.model_params.update(
                    objective = "regression_l2", # mse
                    metric    = "l2",
                )
            # For use in train_model
            self.forecast_loss_func = args.forecast_loss_func

    def train_model(
        self,
        training_mode: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        use_early_stopping = False,
        track_train_metric: bool = False,
        verbose: bool = False,
        callbacks: list = None
        ) -> Tuple[LGBMWrapper, Dict[str, Any]]:
        """
        Trains the LGBM model.

        Args:
            training_mode: The mode of training, either "hpo" or "final_model".
        """
        # Set mode
        if training_mode not in ["hpo", "final_model"]:
            raise ValueError("Mode must be either 'hpo' or 'final_model'.")
        
        # If HPO, single job per trial, as Optuna parallelizes trials
        if training_mode == "hpo":
            self.model_params.update(n_jobs = 1)
        # Use all CPUs for final training:
        elif training_mode == "final_model":
            self.model_params.update(n_jobs = -1)

        # Initialize the model
        model = LGBMWrapper(**self.model_params)

        # Dynamically build the evaluation set
        eval_set = []
        eval_names = []
        
        # If we want to track the training metric, we add the training set to eval_set
        if track_train_metric:
            eval_set.append((X_train, y_train))
            eval_names.append('train')
        
        # If validation data is provided, then add it
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append('valid')
        
        # Fit the model
        model.fit(
            X_train, y_train, 
            eval_set=eval_set if eval_set else None,
            eval_names=eval_names if eval_names else None,
            use_early_stopping=use_early_stopping, 
            verbose=verbose,
            callbacks=callbacks)
        
        # Extract history
        # Instantiate training history
        history = TrainingHistory.from_lgbm(
            model.evals_result_,
            train_metric    = "logloss" if self.args.task_type == "workhour_classification" else self.forecast_loss_func,
            train_objective = "minimize",
            optuna_metric   = ("auc" if self.args.task_type == "workhour_classification" else self.forecast_loss_func) if training_mode=="hpo" else None,
            optuna_objective= ("maximize" if self.args.task_type == "workhour_classification" else "minimize") if training_mode=="hpo" else None,
        )

        training_result = None
        if X_val is not None:
            best_score = history.get_best_valid_score()
            optimal_threshold = None
            if self.args.task_type == "workhour_classification":
                probs_val = model.predict_proba(X_val)
                optimal_threshold = find_optimal_f1_threshold(y_val, probs_val)
                logger.info("Optimal F1 threshold on validation set: %.4f", optimal_threshold)
            
            training_result = TrainingResult(
                metric         = best_score,
                best_epoch     = model.best_iteration_,
                optimal_threshold = optimal_threshold,
            )
        return model, history, training_result
                
    def evaluate_model(
        self,
        model: LGBMWrapper,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_source_df: pd.Series = None,
        threshold: float = 0.5
        ) -> Dict[str, Any]:
        """
        Evaluates the trained model on a test set.
        
        Args:
            model: The trained LGBMWrapper instance.
            X_test: Test features.
            y_test: Test target.
            y_source_df: DataFrame for reconstructing delta predictions (optional).
            threshold: Classification threshold (for classification tasks).
        
        Returns:
            A dictionary of evaluation metrics.
        """
        # Classification
        if self.args.task_type == "workhour_classification":
            probs = model.predict_proba(X_test)
            metrics = binary_classification_results(y_test.values, probs, threshold)
            model_outputs = {
                "probabilities": probs,
                "predictions": (probs >= threshold).astype(int),
                "labels": y_test.values,
            }
            return metrics, model_outputs
        
        # Forecasting
        preds = model.predict(X_test)
        targets = y_test.values

        if self.args.prediction_type == "delta":
            if y_source_df is None:
                raise ValueError("'y_source_df' must be provided when prediction_type='delta'.")
            logger.info("Reconstructing absolute values from delta predictions.")
            preds = y_source_df.values + preds
            targets = y_source_df.values + targets
        
        metrics = regression_results(targets, preds)
        model_outputs = {
            "predictions": preds,
            "targets": targets,
        }
        return metrics, model_outputs