import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from ...config.args import parse_args
from ...utils.filename_util import get_data_filename
from ...utils.block_split import StratifiedBlockSplitter
from ...utils.train_utils import ResultHandler
from .model import LGBMWrapper

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    log_loss, 
    precision_score, 
    recall_score
)

# Logging setup
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def find_optimal_threshold_lgbm(model: LGBMWrapper, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Finds the optimal classification threshold that maximizes F1-score on the validation set.
    """
    probs = model.predict_proba(X_val)
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    
    # Calculate F1 score for each threshold
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 
                 for p, r in zip(precisions[:-1], recalls[:-1])]
    
    if not f1_scores:
        return 0.5 # Fallback if no valid scores are generated

    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]


class LGBMSingleRunner:
    """
    Orchestrates a single machine learning run for a LightGBM model with fixed
    hyperparameters. It trains, evaluates on a validation set for early stopping,
    retrains on the full training data, and finally evaluates on the test set.
    """
    def __init__(self, args: Any):
        self.args = args
        self.run_id = args.experiment_id # Use the same arg for consistency
        self.seed = args.seed + self.run_id
        
        self.input_dict: Dict[str, Any] = {}
        self.load_data_file()

        self.output_dir = args.output_dir
        logger.info(f"Single run outputs will be saved in: {self.output_dir}")

    def load_data_file(self):
        """Loads the pre-computed tabular DataFrame and metadata."""
        fname = get_data_filename(self.args)
        path = os.path.join(self.args.data_dir, "processed", fname)
        
        logger.info(f"Loading pre-computed tabular input from {path}")
        self.input_dict = joblib.load(path)
        logger.info(f"Loaded DataFrame with shape: {self.input_dict['df'].shape}")
        # Assuming reduce_mem_usage is available if needed
        # self.input_dict['df'] = self.reduce_mem_usage(self.input_dict['df'])

    def _get_split_data(self, df, block_ids, blocks):
        """Helper to filter the main DataFrame based on block IDs."""
        indices = []
        for block_id in block_ids:
            indices.extend(blocks[block_id]['bucket_indices'])
        
        split_df = df[df['bucket_idx'].isin(indices)].copy()
        
        target_col = self.input_dict['target_col_name']
        id_cols = ['bucket_idx']
        if self.args.task_type != "measurement_forecast" and 'room_uri' in split_df.columns:
            id_cols.append('room_uri')

        y = split_df[target_col]
        X = split_df.drop(columns=[target_col] + id_cols)
        
        return X, y

    def run(self):
        """Executes the full pipeline for a single run."""
        logger.info(f"===== Starting LGBM Single Run [{self.run_id+1}] | Seed: {self.seed} =====")

        # 1. Initialize splitter and create train/test split
        splitter = StratifiedBlockSplitter(output_dir=self.output_dir, blocks=self.input_dict['blocks'], stratum_size=self.args.stratum_size, seed=self.seed)
        splitter.get_train_test_split()

        # 2. Get a single train/validation fold from the main training set
        train_fold_ids, val_fold_ids = splitter.get_single_split()

        X_train_fold, y_train_fold = self._get_split_data(self.input_dict['df'], train_fold_ids, self.input_dict['blocks'])
        X_val, y_val = self._get_split_data(self.input_dict['df'], val_fold_ids, self.input_dict['blocks'])

        # 3. Define FIXED hyperparameters
        # These are sensible defaults. You can adjust them as needed.
        params = {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 40,
            "max_depth": 7,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "n_jobs": -1,
            "random_state": self.seed
        }

        if self.args.task_type == "workhour_classification":
            params["objective"] = "binary"
            params["metric"] = "auc"
            params["is_unbalance"] = True
        else:
            params["objective"] = "regression_l1"
            params["metric"] = "mae"

        # 4. Train a model with early stopping to find the best iteration
        logger.info("Training with early stopping to find best iteration...")
        temp_model = LGBMWrapper(**params)
        temp_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val, y_val)], use_early_stopping=True, verbose=True)

        best_iteration = temp_model.best_iteration_
        logger.info(f"Found best iteration after {best_iteration} rounds.")

        # For classification, find the optimal threshold on the validation set
        optimal_threshold = 0.5
        if self.args.task_type == "workhour_classification":
            optimal_threshold = find_optimal_threshold_lgbm(temp_model, X_val, y_val)
            logger.info(f"Found optimal threshold: {optimal_threshold:.4f}")

        # 5. Retrain final model on the FULL training data
        logger.info("Retraining final model on the full training dataset...")
        X_train_full, y_train_full = self._get_split_data(self.input_dict['df'], splitter.train_block_ids, self.input_dict['blocks'])
        X_test, y_test = self._get_split_data(self.input_dict['df'], splitter.test_block_ids, self.input_dict['blocks'])

        final_params = params.copy()
        final_params["n_estimators"] = best_iteration
        final_params["metric"] = "logloss" if self.args.task_type == "workhour_classification" else "mae"

        final_model = LGBMWrapper(**final_params)
        final_model.fit(
            X_train_full, y_train_full, 
            eval_set=[(X_train_full, y_train_full)], 
            eval_names=['train'], 
            use_early_stopping=False, 
            verbose=True
        )

        # 6. Evaluate on test set and save results
        logger.info("Evaluating final model on the hold-out test set...")
        
        # Prepare training history for saving
        metric_name = list(final_model.evals_result_['train'].keys())[0] 
        history = {"train_loss": final_model.evals_result_['train'][metric_name], "val_loss": None}

        # Calculate test metrics
        if self.args.task_type == "workhour_classification":
            preds_proba = final_model.predict_proba(X_test)
            preds_label = (preds_proba > optimal_threshold).astype(int)
            
            # Calculate all required metrics
            test_metrics = {
                'accuracy': (preds_label == y_test).mean(),
                'balanced_accuracy': balanced_accuracy_score(y_test, preds_label),
                'precision': precision_score(y_test, preds_label),
                'recall': recall_score(y_test, preds_label),
                'f1': f1_score(y_test, preds_label),
                'roc_auc': roc_auc_score(y_test, preds_proba),
                'auc_pr': average_precision_score(y_test, preds_proba),
                'test_loss': log_loss(y_test, preds_proba),
                'threshold': optimal_threshold,
                'predictions': preds_label, 
                'probabilities': preds_proba, 
                'labels': y_test.values
            }
        else: # Forecasting
            preds = final_model.predict(X_test)
            
            test_loss_mse = mean_squared_error(y_test, preds)

            # Safely calculate MAPE, avoiding division by zero
            y_test_np = y_test.values # Ensure it's a numpy array for boolean indexing
            mape_mask = y_test_np != 0
            if np.sum(mape_mask) > 0:
                mape = np.mean(np.abs((y_test_np[mape_mask] - preds[mape_mask]) / y_test_np[mape_mask])) * 100
            else:
                mape = 0.0 # Assign 0 if all target values are zero

            # Calculating other metrics
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            test_metrics = {
                'test_loss': test_loss_mse,
                'mse':mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'predictions': preds,
                'targets': y_test.values
            }
                
        # Use ResultHandler to save artifacts
        handler = ResultHandler(output_dir=self.output_dir, task_type=self.args.task_type,
                                history=history, metrics=test_metrics, model=final_model)
        handler.process()
        
        # Save scalar metrics to a CSV for easy comparison
        scalar_metrics = {k: v for k, v in test_metrics.items() if np.isscalar(v)}
        scalar_metrics['run_id'] = self.run_id
        scalar_metrics['best_iteration'] = best_iteration
        pd.DataFrame([scalar_metrics]).to_csv(os.path.join(self.output_dir, "results_test_single.csv"), index=False)

        logger.info(f"===== Single Run [{self.run_id+1}] COMPLETED. =====")


def main():
    # Assuming parse_args() is defined in your config module
    args = parse_args()
    
    runner = LGBMSingleRunner(args)
    runner.run()

if __name__ == '__main__':
    main()