import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

import optuna
from optuna.pruners import MedianPruner
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
    root_mean_squared_error,
    r2_score)

from ...utils.filename_util import get_data_filename
from ...utils.block_split import StratifiedBlockSplitter
from ...utils.train_utils import ResultHandler
from .model import LGBMWrapper

# Set up the main logging
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class LGBMExperimentRunner:
    """
    Orchestrates a machine learning experiment for a LightGBM model, designed
    to be run in parallel for different random seeds.
    """
    def __init__(self, args: Any):
        self.args = args

        # A unique ID for this specific experiment run (e.g., from SLURM_ARRAY_TASK_ID)
        self.experiment_id = args.experiment_id
        # Create a unique seed for this run to ensure data splits are different
        self.seed = args.seed + args.experiment_id

        # Pruning
        self.median_pruning = {
            "n_startup_trials": args.n_startup_trials,
            "n_warmup_steps": args.n_warmup_steps,
            "interval_steps": args.interval_steps
        }

        # Central lists to store all detailed records
        self.cv_records: List[Dict[str, Any]] = []
        self.test_records: List[Dict[str, Any]] = []

        # Pre-loading the main data file into memory
        self.input_dict: Dict[str, Any] = {}
        self.load_data_file()

        # Set up the specific output directory for this run's artifacts
        self.output_dir = args.output_dir
        logger.info(f"Experiment outputs will be saved in: {self.output_dir}")

    def load_data_file(self):
        """Loads the pre-computed tabular DataFrame and metadata."""
        fname = get_data_filename(self.args)
        path = os.path.join(self.args.data_dir, "processed", fname)
        
        logger.info(f"Loading pre-computed tabular input from {path}")
        self.input_dict = joblib.load(path)
        logger.info(f"Loaded DataFrame with shape: {self.input_dict['df'].shape}")
        
    def _get_split_data(self, df, block_ids, blocks):
        """Helper to filter the main DataFrame based on block IDs."""
        indices = []
        for block_id in block_ids:
            indices.extend(blocks[block_id]['bucket_indices'])
        
        # Filter based on the 'bucket_idx'
        split_df = df[df['bucket_idx'].isin(indices)].copy()
        
        # Define features (X) and target (y)
        target_col = self.input_dict['target_col_name']
        
        # Drop identifiers and the target column
        # NOTE: For measurement_forecast, 'room_uri' is a feature!
        id_cols = ['bucket_idx']
        if self.args.task_type != "measurement_forecast" and 'room_uri' in split_df.columns:
            id_cols.append('room_uri')

        y = split_df[target_col]
        X = split_df.drop(columns=[target_col] + id_cols)
        
        return X, y

    def run_experiment(self):
        """Executes the full pipeline for a single experiment instance."""
        logger.info(f"===== Starting LGBM Experiment [{self.experiment_id+1}/{self.args.n_experiments}] | Seed: {self.seed} =====")

        splitter = StratifiedBlockSplitter(output_dir=self.output_dir, blocks=self.input_dict['blocks'], stratum_size=self.args.stratum_size, seed=self.seed)
        splitter.get_train_test_split()

        study = self._run_hyperparameter_study(splitter)
        best_trial = study.best_trial
        best_params = best_trial.params

        logger.info(f"Best trial for Experiment [{self.experiment_id+1}/{self.args.n_experiments}]: "
                    f"Metric={best_trial.value:.4f}, Params={best_params}")

        final_model, test_metrics, final_history = self._train_and_eval_final_model(best_params, best_trial, splitter)
        
        handler = ResultHandler(output_dir=self.output_dir, task_type=self.args.task_type,
                                history=final_history, metrics=test_metrics, model=final_model)
        handler.process()
        
        pd.DataFrame(self.cv_records).to_csv(os.path.join(self.output_dir, "results_CV.csv"), index=False)
        
        scalar_metrics = {k: v for k, v in test_metrics.items() if np.isscalar(v)}
        scalar_metrics['experiment_id'] = self.experiment_id
        pd.DataFrame([scalar_metrics]).to_csv(os.path.join(self.output_dir, "results_test.csv"), index=False)

        logger.info(f"===== Experiment [{self.experiment_id+1}/{self.args.n_experiments}] COMPLETED. =====")

    def _run_hyperparameter_study(self, splitter: StratifiedBlockSplitter):
        """Sets up and runs an Optuna study for HPO."""
        direction = "maximize" if self.args.task_type == "workhour_classification" else "minimize"
        
        db_path = os.path.join(self.output_dir, "optuna_study.db")
        storage = f"sqlite:///{db_path}"
        
        study = optuna.create_study(direction=direction, storage=storage, study_name=f"experiment_{self.experiment_id}", 
                                    load_if_exists=True, pruner=MedianPruner())
        
        splitter.get_cv_splits()
        
        # Use n_jobs=-1 to run trials in parallel, utilizing all CPUs from Slurm
        study.optimize(
            lambda trial: self._objective(trial, splitter),
            n_trials=self.args.n_optuna_trials,
            n_jobs=-1, # <<< PARALLEL HPO
            catch=(Exception,)
        )
        return study

    def _objective(self, trial: optuna.trial.Trial, splitter: StratifiedBlockSplitter):
        """The objective function for Optuna, performing k-fold cross-validation."""
        # Suggest hyperparameters for LightGBM
        params = {
            "n_estimators": 1500, # High value, rely on early stopping
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "n_jobs": 1 # Single job per trial, as Optuna parallelizes trials
        }

        if self.args.task_type == "workhour_classification":
            params["objective"] = "binary"
            params["metric"] = "auc" # Optimize for AUC
            params["is_unbalance"] = True
        else: # Forecasting tasks
            params["objective"] = "regression_l1" # MAE
            params["metric"] = "mae"

        fold_metrics = []
        fold_best_iterations = []
        fold_optimal_thresholds = []
        for fold_num, (train_ids, val_ids) in enumerate(splitter.split()):
            X_train, y_train = self._get_split_data(self.input_dict['df'], train_ids, self.input_dict['blocks'])
            X_val, y_val = self._get_split_data(self.input_dict['df'], val_ids, self.input_dict['blocks'])

            model = LGBMWrapper(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            metric_name = list(model.evals_result_['valid_0'].keys())[0]
            metric_val = model.evals_result_['valid_0'][metric_name][model.best_iteration_ - 1]
            
            fold_metrics.append(metric_val)
            fold_best_iterations.append(model.best_iteration_)

            if self.args.task_type == "workhour_classification":
                threshold = find_optimal_threshold_lgbm(model, X_val, y_val)
                fold_optimal_thresholds.append(threshold)

            # Pruning callback for Optuna
            trial.report(metric_val, fold_num)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        avg_metric = np.mean(fold_metrics)
        avg_iterations = np.mean(fold_best_iterations)
        trial.set_user_attr("average_best_iteration", avg_iterations)

        if self.args.task_type == "workhour_classification":
            avg_threshold = np.mean(fold_optimal_thresholds)
            trial.set_user_attr("optimal_threshold", avg_threshold)

        self.cv_records.append({'trial': trial.number, 'metric': avg_metric, **trial.params})

        return avg_metric

    def _train_and_eval_final_model(self, best_params: Dict, best_trial: optuna.trial.FrozenTrial, splitter: StratifiedBlockSplitter):
        """Trains the final model and evaluates it on the test set."""
        logger.info("Retraining final model on the full training dataset...")
        
        train_ids = splitter.train_block_ids
        test_ids = splitter.test_block_ids

        X_train, y_train = self._get_split_data(self.input_dict['df'], train_ids, self.input_dict['blocks'])
        X_test, y_test = self._get_split_data(self.input_dict['df'], test_ids, self.input_dict['blocks'])

        # --- Determine the optimal n_estimators from the HPO study ---
        avg_iter_from_cv = best_trial.user_attrs.get("average_best_iteration")
        if avg_iter_from_cv is None:
            logger.warning("Could not find 'average_best_iteration' in trial. Defaulting to 1000 estimators.")
            final_n_estimators = 1000
        else:
            # Heuristic: Train for slightly longer on the full dataset
            final_n_estimators = int(np.ceil(avg_iter_from_cv * 1.1))
            logger.info(f"Inferred optimal n_estimators from CV: {avg_iter_from_cv:.0f}. "
                        f"Training final model for {final_n_estimators} rounds (1.1x).")

        # Prepare final model parameters
        final_params = best_params.copy()
        final_params["n_estimators"] = final_n_estimators
        final_params.pop("early_stopping_rounds", None) 
        final_params["n_jobs"] = -1 # Use all CPUs for final training
        
        if self.args.task_type == "workhour_classification":
            final_params["objective"] = "binary"
            final_params["metric"] = "logloss"
            final_params["is_unbalance"] = True
        else:
            final_params["objective"] = "regression_l1"
            final_params["metric"] = "mae"

        final_model = LGBMWrapper(**final_params)

        # Fit the model, telling it to track metrics on the training set
        # NO early stopping here
        final_model.fit(X_train, y_train, 
                        eval_set=[(X_train, y_train)], 
                        eval_names=['train'], 
                        verbose=False)
        
        # --- Evaluation on the unseen test set ---
        logger.info("Evaluating final model on the hold-out test set...")

        # Saving train loss history
        raw_history = final_model.evals_result_
        # Dynamically get the metric name ('mae', 'logloss', etc.)
        metric_name = list(raw_history['train'].keys())[0] 
        history = {
            "train_loss": raw_history['train'][metric_name],
            "val_loss": None
        }

        # Metrics
        metrics = {}
        if self.args.task_type == "workhour_classification":
            optimal_threshold = best_trial.user_attrs.get("optimal_threshold", 0.5)
            logger.info(f"Using optimal classification threshold for evaluation: {optimal_threshold:.4f}")

            preds_proba = final_model.predict_proba(X_test)
            preds_label = (preds_proba > optimal_threshold).astype(int)

            metrics = {
                'accuracy': (preds_label == y_test).mean(),
                'balanced_accuracy': balanced_accuracy_score(y_test, preds_label),
                'f1': f1_score(y_test, preds_label),
                'roc_auc': roc_auc_score(y_test, preds_proba),
                'auc_pr': average_precision_score(y_test, preds_proba),
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

            metrics = {
                'test_loss': test_loss_mse,
                'rmse': root_mean_squared_error(y_test, preds),
                'mae': mean_absolute_error(y_test, preds),
                'r2': r2_score(y_test, preds),
                'mape': mape,
                'predictions': preds,
                'targets': y_test.values
            }
        
        return final_model, metrics, history

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
        return 0.5 # Fallback if no valid scores are generated

    # Find the threshold that gives the best F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


def main():
    from ...config.args import parse_args
    args = parse_args()
    
    runner = LGBMExperimentRunner(args)
    runner.run_experiment()

if __name__ == '__main__':
    main()

