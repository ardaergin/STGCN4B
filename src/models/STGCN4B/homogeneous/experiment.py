#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import optuna
from copy import deepcopy
from typing import Dict, Any, List, Tuple

from ....utils.block_split import StratifiedBlockSplitter
from .graph_loader import load_data
from .train import setup_model, train_model, evaluate_model, find_optimal_threshold

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress Optuna's INFO messages to keep the logs cleaner
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ExperimentRunner:
    """
    Orchestrates a full machine learning experiment with nested cross-validation.
    
    The runner performs two loops:
    1. An outer loop that creates multiple, independent train-test splits to
       assess model robustness and generalization.
    2. An inner loop that, for each train-test split, performs k-fold
       cross-validation on the training data to find the best hyperparameters
       using Optuna.

    After finding the best hyperparameters, the model is retrained on the full
    training set and evaluated on the hold-out test set. Results from all
    outer loops are aggregated to provide a final performance estimate.
    """

    def __init__(self, args: Any):
        """
        Initializes the ExperimentRunner.

        Args:
            args: A configuration object (e.g., from argparse) containing all
                  necessary parameters for data loading, model setup, and training.
                  Must include `n_outer_splits` and `n_optuna_trials`.
        """
        self.args = args
        self.outer_loop_results: List[Dict[str, Any]] = []

        # Pre-loading the main data file into memory
        self.input_dict: Dict[str, Any] = {}
        self.load_data_file()

    def load_data_file(self):
        """
        Loads the main data file from disk and prepares all data tensors,
        moving them to the correct device. This is done only once.
        """
        # Save args for easy access
        args = self.args

        # 1) Load saved torch input for homogeneous graph
        if args.task_type == "measurement_forecast":
            fname = f"torch_input_{args.adjacency_type}_{args.interval}_{args.graph_type}_{args.measurement_type}.pt"
        else:
            fname = f"torch_input_{args.adjacency_type}_{args.interval}_{args.graph_type}.pt"
        path = os.path.join(args.data_dir, "processed", fname)
        logger.info(f"Loading homogeneous STGCN input from {path}")
        torch_input = torch.load(path, map_location="cpu")

        # 2) Device setup
        device = (
            torch.device("cuda")
            if args.enable_cuda and torch.cuda.is_available()
            else torch.device("cpu")
        )
        logger.info(f"Using device: {device}")

        # 3) Move adjacency matrices to device
        torch_input["adjacency_matrix"] = torch_input["adjacency_matrix"].to(device)
        if torch_input.get("dynamic_adjacencies") is not None:
            for step, mat in torch_input["dynamic_adjacencies"].items():
                torch_input["dynamic_adjacencies"][step] = mat.to(device)

        # 4) Move feature matrices to device
        #    Each entry torch_input["feature_matrices"][t] is a tensor (R×F)
        for t, mat in torch_input["feature_matrices"].items():
            torch_input["feature_matrices"][t] = mat.to(device)

        # 5A) Select targets based on task, move to device
        if args.task_type == "measurement_forecast":
            targets = torch_input["measurement_values"].to(device)  # shape (T, N)
        elif args.task_type == "consumption_forecast":
            targets = torch_input["consumption_values"].to(device) # shape (T, 1)
        elif args.task_type == "workhour_classification":
            targets = torch_input["workhour_labels"].to(device) # shape (T, )

        # 5B) Mask creation
        mask = torch_input.get("measurement_mask") 
        if mask is None:
            logger.info("No measurement mask found. Creating a default mask of all ones.")
            mask = torch.ones_like(targets, dtype=torch.float32)
        mask = mask.to(device)

        # 10) Return everything downstream might need
        self.input_dict = {
            "device": device,
            # Main data
            "targets": targets,
            "feature_matrices": torch_input["feature_matrices"],
            "mask": mask,
            # Data indices
            "blocks": torch_input["blocks"],
            "time_buckets": torch_input["time_buckets"],
            # Adjacency and dynamic adjacency
            "adjacency_matrix": torch_input["adjacency_matrix"],
            "dynamic_adjacencies": torch_input.get("dynamic_adjacencies", None),
            # Feature metadata
            "room_uris": torch_input["room_uris"],
            "n_nodes": len(torch_input["room_uris"]),
            # Features
            "n_features": torch_input["n_features"], # Total features: static + temporal
            "static_feature_count": torch_input["static_feature_count"],
            "temporal_feature_count": torch_input["temporal_feature_count"],
            "feature_names": torch_input["feature_names"],
        }

        logger.info("STGCN data loaded")

        return None
    
    def get_data_loaders(self, 
                        train_block_ids: List[int],
                        val_block_ids:   List[int],
                        test_block_ids:  List[int]):
        """Wrapper to call the lean `load_data` function with pre-loaded data."""
        return load_data(
            self.args,
            blocks=self.input_dict["blocks"],
            feature_matrices=self.input_dict["feature_matrices"],
            targets=self.input_dict["targets"],
            mask=self.input_dict["mask"],
            train_block_ids=train_block_ids,
            val_block_ids=val_block_ids,
            test_block_ids=test_block_ids)


    def run_experiment(self):
        """Executes the entire experimental pipeline."""
        logger.info("Starting experiment orchestration...")
        all_blocks = self.input_dict["blocks"]

        ########## Outer Loop: Multiple Train-Test Splits ##########
        for i in range(self.args.n_outer_splits):
            seed = self.args.seed + i
            logger.info(f"--- Starting Outer Loop Iteration {i+1}/{self.args.n_outer_splits} (Seed: {seed}) ---")

            # 1. Create a new splitter with a new seed for a fresh train-test split.
            splitter = StratifiedBlockSplitter(all_blocks, stratum_size=self.args.stratum_size, seed=seed)
            splitter.get_train_test_split()

            # 2. Run the inner loop: hyperparameter tuning with CV.
            study = self._run_hyperparameter_study(splitter, seed)
            best_trial = study.best_trial
            best_params = best_trial.params
            best_params['epochs'] = int(np.ceil(best_trial.user_attrs["best_n_epochs"])) # the optimal number of epochs determined during CV

            logger.info(f"Best trial for outer loop {i+1}: Metric={best_trial.value:.4f}, Params={best_params}")

            # 3. Retrain the final model on the entire training set with the best hyperparameters.
            final_model, test_loader = self._train_final_model(best_params, splitter.train_block_ids, splitter.test_block_ids)

            # 4. Evaluate the retrained model on the hold-out test set.
            test_metrics = self._evaluate_final_model(final_model, test_loader, best_trial)
            self.outer_loop_results.append(test_metrics)
            
            # Log metrics, filtering out non-scalar values for clarity
            scalar_metrics = {k: v for k, v in test_metrics.items() if np.isscalar(v)}
            logger.info(f"Test metrics for outer loop {i+1}: {scalar_metrics}")

        # 5. Aggregate and report final results across all outer loop iterations.
        self._report_final_results()

    def _run_hyperparameter_study(self, splitter, seed):
        """Sets up and runs an Optuna study for hyperparameter optimization."""
        direction = "maximize" if self.args.task_type == "workhour_classification" else "minimize"
        study = optuna.create_study(direction=direction)
        objective_func = lambda trial: self._objective(trial, splitter)
        study.optimize(objective_func, n_trials=self.args.n_optuna_trials)
        return study
    
    def _objective(self, trial, splitter):
        """The objective function for Optuna, performing k-fold cross-validation."""
        splitter.get_cv_splits()
        trial_args = deepcopy(self.args)

        # --- General Training Hyperparameters ---
        trial_args.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        trial_args.weight_decay_rate = trial.suggest_float("weight_decay_rate", 1e-5, 1e-4, log=True)
        trial_args.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])

        # --- STGCN Architecture Hyperparameters ---
        trial_args.n_his = trial.suggest_int("n_his", 12, 24, step=12) # e.g., 12 or 24 hours
        trial_args.stblock_num = trial.suggest_int("stblock_num", 2, 3)
        trial_args.Kt = trial.suggest_categorical("Kt", [2, 3])
        trial_args.Ks = trial.suggest_categorical("Ks", [2, 3])
        trial_args.act_func = trial.suggest_categorical("act_func", ["glu", "relu"])
        trial_args.droprate = trial.suggest_float("droprate", 0.3, 0.5)

        # We don't tune epochs directly. We set a max value and let early stopping find the best.
        # This is the max number of epochs the model is allowed to run for in each CV fold.
        trial_args.epochs = 300

        # Loop through CV folds.
        fold_metrics, fold_actual_epochs, fold_thresholds = [], [], []
        for train_ids, val_ids in splitter.split():
            # Load data for the current fold, combine loaders with other necessary data for setup_model
            loaders = self.get_data_loaders(train_ids, val_ids, test_block_ids=[])
            data_for_setup = {**self.input_dict, **loaders}

            model, criterion, optimizer, scheduler, early_stopping = setup_model(trial_args, data_for_setup)
            _, history = train_model(trial_args, model, criterion, optimizer, scheduler, 
                                    early_stopping=early_stopping, 
                                    train_loader=loaders['train_loader'], 
                                    val_loader=loaders['val_loader'])
            
            fold_actual_epochs.append(early_stopping.best_epoch)

            if self.args.task_type == "workhour_classification":
                fold_metrics.append(max(history['val_metrics']['f1']))
                # Reload the best model state to find the threshold that produced it
                model.load_state_dict(early_stopping.best_model_state)
                fold_thresholds.append(find_optimal_threshold(model, loaders['val_loader']))
            else: # Forecasting tasks
                fold_metrics.append(min(history['val_loss']))

        # Aggregate results and save to trial
        avg_metric = np.mean(fold_metrics)
        trial.set_user_attr("best_n_epochs", np.mean(fold_actual_epochs))
        if self.args.task_type == "workhour_classification":
            trial.set_user_attr("optimal_threshold", np.mean(fold_thresholds))
            
        return avg_metric

    def _train_final_model(self, best_params, train_ids, test_ids):
        """Trains the final model on the full training set."""
        logger.info("Retraining final model on all available training data...")

        final_args = deepcopy(self.args)
        for key, value in best_params.items():
            setattr(final_args, key, value)
        
        loaders = self.get_data_loaders(train_ids, val_block_ids=[], test_block_ids=test_ids)
        data_for_setup = {**self.input_dict, **loaders}
        
        model, criterion, optimizer, scheduler, _ = setup_model(final_args, data_for_setup)
        final_model, _ = train_model(final_args, model, criterion, optimizer, scheduler, 
                                     early_stopping=None, 
                                     train_loader=loaders['train_loader'], 
                                     val_loader=None)
        return final_model, loaders['test_loader']

    def _evaluate_final_model(self, model: torch.nn.Module, test_loader: DataLoader, best_trial: optuna.trial.FrozenTrial):
        """Evaluates the final model on the hold-out test set."""
        if self.args.task_type == "workhour_classification":
            # Retrieve the optimal threshold found during CV for the best trial.
            # Use 0.5 as a safe fallback if it's not found.
            optimal_threshold = best_trial.user_attrs.get("optimal_threshold", 0.5)
            logger.info(f"Using optimal classification threshold for evaluation: {optimal_threshold:.4f}")
            
            # Pass this specific threshold to the evaluation function.
            metrics = evaluate_model(self.args, model, test_loader, threshold=optimal_threshold)
        else:
            # For forecasting, no threshold is needed.
            metrics = evaluate_model(self.args, model, test_loader)
        return metrics

    def _report_final_results(self):
        """Aggregates and logs the final results from all outer loop iterations."""
        logger.info("---" * 20 + "\n--- Final Aggregated Experiment Results ---\n" + "---" * 20)
        if not self.outer_loop_results:
            logger.warning("No results were recorded.")
            return

        results_df = pd.DataFrame(self.outer_loop_results)
        scalar_cols = [c for c, dtype in results_df.dtypes.items() if 'object' not in str(dtype)]
        
        if not scalar_cols:
            logger.warning("No scalar metrics found to report.")
            return
            
        scalar_results_df = results_df[scalar_cols]
        logger.info("Results across all outer splits:\n" + scalar_results_df.to_string())
        mean_metrics = scalar_results_df.mean()
        std_metrics = scalar_results_df.std()
        summary_df = pd.DataFrame({'mean': mean_metrics, 'std': std_metrics})
        logger.info("\n--- Aggregated Metrics (Mean ± Std) ---\n" + summary_df.to_string(float_format="%.4f") + "\n" + "---" * 20)
