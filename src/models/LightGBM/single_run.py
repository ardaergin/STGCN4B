#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay)
# Regression Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

from ...utils.filename_util import get_data_filename
from ...utils.train_utils import ResultHandler
from ...preparation.split import StratifiedBlockSplitter
from ...preparation.preparer import LGBMDataPreparer
from .model import LGBMWrapper

# Set up the main logging
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


class LGBMSingleRunner:
    """
    Runs a single LightGBM experiment with default parameters.
    No Optuna optimization or cross-validation.
    """
    
    def __init__(self, args: Any):
        """
        Initializes the LGBMSingleRunner for a single experiment instance.

        Args:
            args: A configuration object (e.g., from argparse) containing all
                  necessary parameters.
        """
        self.args = args
        self.seed = args.seed

        # Set up the specific output directory for this run's artifacts
        self.output_dir = args.output_dir
        logger.info(f"Experiment outputs will be saved in: {self.output_dir}")

        # The preparer returns everything we need for the experiment
        logger.info("Handling data preparation...")
        data_preparer = LGBMDataPreparer(args)
        self.input_dict = data_preparer.get_input_dict()
        self.delta_to_absolute_map = self.input_dict.get("delta_to_absolute_map", {})

    def _get_split_data(self, block_ids: List[int]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Helper to filter the final DataFrame based on block IDs."""
        indices = []
        blocks = self.input_dict['blocks']
        for block_id in block_ids:
            indices.extend(blocks[block_id]['bucket_indices'])
        
        # Filter based on the 'bucket_idx'
        split_df = self.input_dict['df'][self.input_dict['df']['bucket_idx'].isin(indices)].copy()
        
        # Get target (y)
        target_colnames = self.input_dict['target_colnames']
        if len(target_colnames) != 1:
            raise ValueError(f"This script is intended for a single target, but found {len(target_colnames)}: {target_colnames}")
        target_col_name = target_colnames[0]
        y = split_df[target_col_name]

        # Delta forecasting logic
        reconstruction_t_df = None
        if self.args.prediction_type == "delta":
            reconstruction_t_df = split_df[self.input_dict["source_colname"]].copy()
        
        # Get features (X)
        cols_to_drop = ['bucket_idx'] + target_colnames + self.input_dict.get("delta_colnames", [])
        cols_to_drop = [col for col in cols_to_drop if col in split_df.columns]
        X = split_df.drop(columns=cols_to_drop)
        
        return X, y, reconstruction_t_df
            
    def run_experiment(self):
        """Executes the single run pipeline."""
        logger.info(f"===== Starting Single LGBM Experiment | Seed: {self.seed} =====")
        splitter = StratifiedBlockSplitter(output_dir=self.output_dir, blocks=self.input_dict['blocks'], 
                                            stratum_size=self.args.stratum_size, seed=self.seed)
        
        # Get train/validation/test split
        splitter.get_train_test_split()
        train_ids, val_ids = splitter.get_single_split()
        test_ids = splitter.test_block_ids
        # Get default parameters
        default_params = self._get_default_params()
        logger.info(f"Using default parameters: {default_params}")
        # Train model with validation set for early stopping
        model, val_results = self._train_with_validation(default_params, train_ids, val_ids)
        
        # Evaluate on test set, passing the optimal threshold from the validation results
        test_metrics, final_history = self._evaluate_on_test(model, test_ids, val_results)
        
        # Process and save results
        handler = ResultHandler(output_dir=self.output_dir, task_type=self.args.task_type,
                                history=final_history, metrics=test_metrics, model=model)
        handler.process()
        
        # Save test metrics
        scalar_metrics = {k: v for k, v in test_metrics.items() if np.isscalar(v)}
        pd.DataFrame([scalar_metrics]).to_csv(os.path.join(self.output_dir, "results_test.csv"), index=False)
        logger.info(f"===== Single Experiment COMPLETED =====")

    def _train_with_validation(self, params: Dict[str, Any], train_ids: List[int], val_ids: List[int]):
        """Train model with validation set for early stopping."""
        X_train, y_train, _ = self._get_split_data(block_ids=train_ids)
        X_val, y_val, _ = self._get_split_data(block_ids=val_ids)
        model = LGBMWrapper(**params)
        model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=['train', 'valid'],
                use_early_stopping=True, 
                verbose=True)
        
        # Get validation results
        val_results = {}
        if self.args.task_type == "workhour_classification":
            # Find optimal threshold on validation set
            optimal_threshold = find_optimal_threshold_lgbm(model, X_val, y_val)
            val_results['optimal_threshold'] = optimal_threshold
            logger.info(f"Found optimal threshold on validation set: {optimal_threshold:.4f}")
        
        val_results['best_iteration'] = model.best_iteration_
        logger.info(f"Best iteration: {model.best_iteration_}")
        return model, val_results

    def _evaluate_on_test(self, model: LGBMWrapper, test_ids: List[int], val_results: Dict[str, Any]):
        """Evaluate the model on the test set."""
        logger.info("Evaluating model on the hold-out test set...")
        X_test, y_test, reconstruction_t_df_test = self._get_split_data(block_ids=test_ids)

        # Get training history
        raw_history = model.evals_result_

        # Get the validation loss
        val_metric_name = list(raw_history['valid'].keys())[0]
        val_loss = raw_history['valid'][val_metric_name]

        # Get the training loss
        train_metric_name = list(raw_history['train'].keys())[0]
        train_loss = raw_history['train'][train_metric_name]

        history = {
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        # Compute test metrics
        metrics = {}
        if self.args.task_type == "workhour_classification":
            preds_proba = model.predict_proba(X_test)
            
            # Use the optimal threshold found on the validation set
            optimal_threshold = val_results.get('optimal_threshold', 0.5)
            preds_label = (preds_proba > optimal_threshold).astype(int)
            
            metrics = {
                'test_loss': log_loss(y_test, preds_proba),
                'accuracy': accuracy_score(y_test, preds_label),
                'balanced_accuracy': balanced_accuracy_score(y_test, preds_label),
                'precision': precision_score(y_test, preds_label),
                'recall': recall_score(y_test, preds_label),
                'f1': f1_score(y_test, preds_label),
                'roc_auc': roc_auc_score(y_test, preds_proba),
                'auc_pr': average_precision_score(y_test, preds_proba),
                'predictions': preds_label,
                'probabilities': preds_proba,
                'labels': y_test.values,
                'threshold': optimal_threshold # Standardize key name to 'threshold'
            }
        else:  # Forecasting
            preds = model.predict(X_test)
            
            if self.args.prediction_type == "delta":
                logger.info("Reconstructing absolute values from delta predictions for evaluation...")
                # Get the base values at time t (test set)
                values_at_t = reconstruction_t_df_test.values

                # Calculate final absolute predictions: value at t + predicted delta
                preds_final = values_at_t + preds
                
                # Calculate final absolute targets: value at t + actual delta (y_test)
                targets_final = values_at_t + y_test.values

            else:
                # If not in delta mode, preds and targets are already absolute
                preds_final = preds
                targets_final = y_test.values

            test_loss_mse = mean_squared_error(targets_final, preds_final)

            # Safely calculate MAPE, avoiding division by zero
            mape_mask = targets_final != 0
            if np.sum(mape_mask) > 0:
                mape = np.mean(np.abs((targets_final[mape_mask] - preds_final[mape_mask]) / targets_final[mape_mask])) * 100
            else:
                mape = 0.0

            # Calculating other metrics
            mse = mean_squared_error(targets_final, preds_final)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets_final, preds_final)
            r2 = r2_score(targets_final, preds_final)

            metrics = {
                'test_loss': test_loss_mse,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'predictions': preds_final,
                'targets': targets_final
            }
        
        return metrics, history

    def _get_default_params(self) -> Dict[str, Any]:
        """Returns default LightGBM parameters."""
        params = {
            "learning_rate": 0.05,
            "num_leaves": 100,
            "max_depth": 6,
            "min_child_samples": 20,
            "min_child_weight": 0.001,
            "min_split_gain": 0.001,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.01,
            "lambda_l2": 0.01,
            "n_jobs": -1,  # Use all available CPUs
            "n_estimators": self.args.n_estimators,  # Default is 1500
            "early_stopping_rounds": self.args.early_stopping_rounds,
        }

        if self.args.task_type == "workhour_classification":
            params["objective"] = "binary"
            params["metric"] = "auc"
            params["is_unbalance"] = True
        else:  # Forecasting tasks
            params["objective"] = "regression_l1"  # MAE
            params["metric"] = "mae"

        return params


# Threshold Helper
def find_optimal_threshold_lgbm(model: LGBMWrapper, X_val: pd.DataFrame, y_val: pd.Series):
    """
    Finds the optimal classification threshold that maximizes F1-score on the validation set.
    """
    # Get probability predictions for the positive class
    probs = model.predict_proba(X_val)
    
    # Calculate precision, recall, and thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    
    # Calculate F1 score for each threshold, avoiding division by zero
    # Note: thresholds array is one element shorter than precisions/recalls
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 
                 for p, r in zip(precisions[:-1], recalls[:-1])]
    
    if not f1_scores:
        return 0.5  # Fallback if no valid scores are generated

    # Find the threshold that gives the best F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


def main():
    from ...config.args import parse_args
    args = parse_args()
    
    runner = LGBMSingleRunner(args)
    runner.run_experiment()


if __name__ == '__main__':
    main()