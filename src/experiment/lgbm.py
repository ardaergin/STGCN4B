#!/usr/bin/env python

from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
from argparse import Namespace
import gc
import optuna

from ..preparation.preparer import TabularDataPreparer
from ..models.LightGBM.train import LGBMTrainer
from ..utils.tracking import TrainingResult, TrainingHistory
from .base import BaseExperimentRunner

# Set up the main logging
import logging; logger = logging.getLogger(__name__)


class LGBMExperimentRunner(BaseExperimentRunner):
    def __init__(self, args: Any):
        super().__init__(args)
    
    def _prepare_data(self) -> Dict[str, Any]:
        logger.info("Handling data preparation via TabularDataPreparer...")
        data_preparer = TabularDataPreparer(self.args)
        input_dict = data_preparer.get_input_dict()
        return input_dict
        
    #########################
    # Split preparation
    #########################
    
    def _normalize_split(self):
        """No need for normalization in LGBM."""
        pass

    def _impute_split(self):
        """No need for imputation in LGBM."""
        pass

    def _get_split_payload(self, block_ids: List[int]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
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
            raise ValueError(f"This script is intended for a single target, "
                             f"but found {len(target_colnames)}: {target_colnames}")
        target_colname = target_colnames[0]
        y = split_df[target_colname]
        
        # Delta forecasting logic
        target_source_df = None
        if self.args.prediction_type == "delta":
            source_colname = self.input_dict["source_colname"]
            full_target_source_df = self.input_dict["target_source_df"]
            
            # Define the columns to merge on
            merge_on_cols = ['bucket_idx']
            if self.args.task_type == "measurement_forecast":
                merge_on_cols.append('room_uri_str')
            merge_keys_df = split_df[merge_on_cols].copy()
            
            # Merge
            merged_df = pd.merge(
                merge_keys_df, full_target_source_df, 
                on=merge_on_cols, how='left'
            )
            target_source_df = merged_df[source_colname]
        
        # Get features (X)
        cols_to_drop = ['bucket_idx'] + target_colnames
        cols_to_drop = [col for col in cols_to_drop if col in split_df.columns]
        X = split_df.drop(columns=cols_to_drop)
        
        return X, y, target_source_df

    #########################
    # HPO
    #########################

    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """The objective function for Optuna, performing k-fold cross-validation."""
        trial_args = deepcopy(self.args)

        # --- Learning Control & Core Parameters ---
        trial_args.learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
        trial_args.lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True)
        trial_args.lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True)

        # --- Tree Structure ---
        trial_args.num_leaves = trial.suggest_int("num_leaves", 20, 300)
        trial_args.max_depth = trial.suggest_int("max_depth", 3, 12)
        trial_args.min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
        trial_args.min_child_weight = trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True)
        trial_args.min_split_gain = trial.suggest_float("min_split_gain", 1e-4, 1.0, log=True)

        # --- Sampling & Feature Selection ---
        trial_args.feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
        trial_args.bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)
        trial_args.bagging_freq = trial.suggest_int("bagging_freq", 1, 7)

        # --- Static Parameters (not tuned) ---
        # NOTE: Using a high number of estimators (default=1500), 
        #       and relying on early stopping to find the optimal number.
        trial_args.n_estimators = self.args.n_estimators
        trial_args.early_stopping_rounds = self.args.early_stopping_rounds
        
        return trial_args
    
    ##########################
    # Setup model
    ##########################
    def _setup_model(
            self, 
            args: Any,
        ) -> LGBMTrainer:
        """Sets up the LightGBM model trainer."""
        model = LGBMTrainer(args=args)
        return model
    
    ##########################
    # Experiment execution
    ##########################

    def _train_one_fold(
        self,
        model:              Any,
        trial:              optuna.trial.Trial,
        trial_params:       Dict[str, Any],
        fold_index:         int = None,
        *,
        train_block_ids:    List[int],
        val_block_ids:      List[int],
        ) -> TrainingResult:
        """Trains and evaluates a single fold in cross-validation for HPO."""
        
        # 1. Setting up the pruning callback
        metric_to_monitor = "auc" if self.args.task_type == "workhour_classification" else "l1" # mae

        def pruning_callback_with_offset(env: object) -> None:
            """
            A custom callback to report intermediate results to Optuna with an offset.
            'env' is a special object passed by LightGBM at each boosting round.
            """
            # Find the validation metric in the evaluation results
            current_score = None
            for data_name, metric_name, metric_value, _ in env.evaluation_result_list:
                if data_name == "valid" and metric_name == metric_to_monitor:
                    current_score = metric_value
                    break
            if current_score is None:
                return
            # Calculate the global step across all CV folds
            global_step = env.iteration + (fold_index * self.args.n_estimators)
            # Report to Optuna and check for pruning
            trial.report(current_score, global_step)
            if trial.should_prune():
                message = f"Trial was pruned at iteration {env.iteration} in fold {fold_index}."
                raise optuna.exceptions.TrialPruned(message)
        
        # 2. Get split payload
        X_train, y_train, _ = self._get_split_payload(block_ids=train_block_ids)
        X_val, y_val, _ = self._get_split_payload(block_ids=val_block_ids)
        
        # Set the seed
        trial_seed = self.seed + fold_index
        model.model_params.update(
            seed                    = trial_seed,
            bagging_seed            = trial_seed,
            feature_fraction_seed   = trial_seed,
            data_random_seed        = trial_seed,
        )

        # NOTE: We need a try-except block because our callback raises TrialPruned,
        # which we need to catch and let Optuna handle.
        try:
            trained_model, history, training_result = model.train_model(
                training_mode       = "hpo",
                X_train             = X_train, 
                y_train             = y_train, 
                X_val               = X_val,
                y_val               = y_val,
                use_early_stopping  = True,
                track_train_metric  = True,
                verbose             = True,
                callbacks           = [pruning_callback_with_offset]
            )
            return training_result
        except optuna.exceptions.TrialPruned as e:
            logger.warning(f"Trial {trial.number} was pruned at fold {fold_index}.")
            trained_model = None
            raise e
        finally:
            # Store the model for cleanup later
            self._fold_model = trained_model
    
    ##########################
    # Cleanup
    ##########################

    def _cleanup_after_fold(self) -> None:
        model_wrapper = getattr(self, "_fold_model", None)
        if model_wrapper is not None:
            try:
                model_wrapper.model_.booster_.free_raw_data()
            except Exception:
                logger.debug("Could not free raw data", exc_info=True)
        self._fold_model = None
        gc.collect()
    
    def _cleanup_after_trial(self) -> None:
        gc.collect()

    ##########################
    # Final Training & Evaluation
    ##########################

    def _train_and_evaluate_final_model(
        self, 
        model:              Any,
        final_params:       Namespace,
        epochs:             int = None,
        threshold:          float = None,
        *,
        train_block_ids:    List[int],
        val_block_ids:      List[int], # For run_mode="test", we do have a validation set
        test_block_ids:     List[int],
        ) -> Tuple[Any, TrainingHistory, Dict[str, Any], Dict[str, Any]]:
        """Trains and evalutes the final model."""        
        # Expose the epochs to args
        # NOTE. Using a heuristic: train for slightly longer on the full dataset.
        if epochs is not None:
            multiplier = self.args.final_epoch_multiplier 
            final_params.n_estimators = int(np.ceil(epochs * multiplier))
            logger.info(f"Inferred optimal n_estimators from CV: {epochs:.0f}. "
                        f"Training final model for {final_params.n_estimators} rounds ({multiplier}x).")
        
        # Get split payload
        X_train, y_train, _         = self._get_split_payload(block_ids = train_block_ids)
        if val_block_ids:
            X_val, y_val, _         = self._get_split_payload(block_ids = val_block_ids)
        X_test, y_test, y_source_df = self._get_split_payload(block_ids = test_block_ids)

        # Final seed
        # NOTE: For final training, using a different seed,
        #       *2 seems high enough to not clash with the seeds of any previous folds.
        final_seed = self.seed * 2  
        model.model_params.update(
            seed                    = final_seed,
            bagging_seed            = final_seed,
            feature_fraction_seed   = final_seed,
            data_random_seed        = final_seed,
        )

        trained_model, history, training_result = model.train_model(
            training_mode       = "final_model",
            X_train             = X_train, 
            y_train             = y_train, 
            X_val               = X_val if val_block_ids else None, 
            y_val               = y_val if val_block_ids else None, 
            use_early_stopping  = True if val_block_ids else False,
            track_train_metric  = True,
            verbose             = True,
            callbacks           = None # No pruning callback here
        )
        
        # Determining the threshold if not already given (from HPO)
        if self.args.task_type == "workhour_classification":
            if threshold is None:
                threshold = training_result.optimal_threshold
            logger.info(f"Using optimal classification threshold for evaluation: {threshold:.4f}")
                
        # Evaluation
        metrics, model_outputs = model.evaluate_model(
            model               = trained_model, 
            X_test              = X_test, 
            y_test              = y_test,
            y_source_df         = y_source_df, 
            threshold           = threshold
        )
        
        return trained_model, history, metrics, model_outputs

def main():
    from ..config.args import parse_args
    args = parse_args()
    
    runner = LGBMExperimentRunner(args)
    runner.run()

if __name__ == '__main__':
    main()