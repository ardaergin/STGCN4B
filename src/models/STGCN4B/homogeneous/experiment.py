#!/usr/bin/env python

import os
import json
from copy import deepcopy
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner

from ....utils.train_utils import ResultHandler
from ....preparation.split import StratifiedBlockSplitter
from ....preparation.preparer import STGCNDataPreparer
from .normalizer import STGCNNormalizer
from .graph_loader import get_data_loaders
from .train import setup_model, train_model, evaluate_model, find_optimal_threshold

# Set up the main logging
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Suppress Optuna's INFO messages to keep the logs cleaner
optuna.logging.set_verbosity(optuna.logging.WARNING)


class STGCNExperimentRunner:
    """
    Orchestrates a machine learning experiment for a STGCN model, designed
    to be run in parallel for different random seeds.
    """

    def __init__(self, args: Any):
        """
        Initializes the STGCNExperimentRunner for a single experiment instance.

        Args:
            args: A configuration object (e.g., from argparse) containing all
                  necessary parameters. It must include:
                    - `split_id`: A unique integer for this experiment run, used for seeding.
                    - `n_experiments`: The total number of parallel experiments being run.
                    - `seed`: A base random seed.
                    - `n_optuna_trials`: The number of HPO trials to run.
        """
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

        # Set up the specific output directory for this run's artifacts
        self.output_dir = args.output_dir
        logger.info(f"Experiment outputs will be saved in: {self.output_dir}")

        # Save configuration (some config are to be disregarded, as they are not used and just defaults)
        self._save_arguments()

        # The preparer returns everything we need for the experiment
        logger.info("Handling data preparation...")
        data_preparer = STGCNDataPreparer(args)
        self.input_dict = data_preparer.get_input_dict()
    
    def _normalize_data_and_get_dataloaders(
            self, args,
            train_block_ids: List[int], val_block_ids: List[int], test_block_ids: List[int],
            splitter: StratifiedBlockSplitter
            )-> Tuple[Dict, STGCNNormalizer]:
        """
        Handles data normalization for a given fold using the fast NumPy workflow.
        
        Returns the final data loaders and the fitted normalizer.
        """
        device = self.input_dict['device']
        
        # --- 1. Get train indices and slice arrays to fit the processor ---
        train_indices = splitter._get_indices_from_blocks(train_block_ids)
        train_feature_slice = self.input_dict["feature_array"][train_indices]
        train_target_slice = self.input_dict["target_array"][train_indices]
        train_mask_slice = self.input_dict["target_mask"][train_indices] if self.input_dict["target_mask"] is not None else None

        # --- 2. Fit processor, transform full arrays, and impute ---
        processor = STGCNNormalizer()

        ##### X #####
        processor.fit_features(
            train_array=train_feature_slice,
            feature_names=self.input_dict["feature_names"],
            method=args.normalization_method,
            features_to_skip_norm=args.skip_normalization_for
            )
        norm_feature_array = processor.transform_features(full_array=self.input_dict["feature_array"])
        norm_feature_array[np.isnan(norm_feature_array)] = 0.0

        # NOTE: I used to create dictionary of T tensors. 
        #       Now, creating create one large tensor, which is a more efficient bulk transfer to the GPU.
        norm_feature_tensor = torch.from_numpy(norm_feature_array).float().to(device)

        ##### y #####
        if self.args.task_type == "workhour_classification":
            targets_numpy = self.input_dict["target_array"]
        else: # Forecasting tasks
            processor.fit_target(
                train_targets=train_target_slice, 
                train_mask=train_mask_slice,
                method='median'
                )
            targets_numpy = processor.transform_target(targets=self.input_dict["target_array"])

        # NOTE: For measurement forecast task, we impute to target
        #       This is fine, since we will mask these imputations later
        if self.args.task_type == "measurement_forecast":
            targets_numpy[np.isnan(targets_numpy)] = 0.0

        # Converting the processed targets (either normalized or original) to a FloatTensor
        final_targets = torch.from_numpy(targets_numpy).float().to(device)

        # Handling masks
        if self.args.task_type == "measurement_forecast":
            if self.input_dict["target_mask"] is None:
                raise ValueError("Task type is measurement_forecast, but target_mask is None.")
            final_mask = torch.from_numpy(self.input_dict["target_mask"]).float().to(device)
        else: # workhour_classification, consumption_forecast
            final_mask = torch.ones_like(final_targets)
        
        # Handling the target source tensor for delta forecasting
        # NOTE: These tensors are NOT normalized. They are kept in their original scale for reconstruction.
        if self.args.prediction_type == "delta":
            target_source_tensor = torch.from_numpy(self.input_dict["target_source_array"]).float().to(device)
        else:
            # If not in delta mode, create a dummy tensors of zeros with the same shape as targets.
            # This simplifies the data loader's interface.
            target_source_tensor = torch.zeros_like(final_targets)
                
        # --- 5. Get DataLoaders ---
        max_target_offset = max(self.args.forecast_horizons) if self.args.task_type != "workhour_classification" else 1
        loaders = get_data_loaders(
            args,
            blocks=self.input_dict["blocks"],
            block_size=self.input_dict["block_size"],
            feature_tensor=norm_feature_tensor,
            target_tensor=final_targets,
            target_mask_tensor=final_mask,
            target_source_tensor=target_source_tensor,
            max_target_offset=max_target_offset,
            train_block_ids=train_block_ids,
            val_block_ids=val_block_ids,
            test_block_ids=test_block_ids
        )
        return loaders, processor
    
    def run_experiment(self):
        """
        Executes the full pipeline for a single experiment instance.
        """
        logger.info("Starting a single split experiment run...")
        all_blocks = self.input_dict["blocks"]
        experiment_id = self.experiment_id
        seed = self.seed

        logger.info(f"===== Starting Experiment [{experiment_id+1}/{self.args.n_experiments}] | Seed: {seed} =====")

        # 1. Create a unique, seeded train-test split for this experiment.
        splitter = StratifiedBlockSplitter(output_dir=self.output_dir, blocks=all_blocks, stratum_size=self.args.stratum_size, seed=seed)
        splitter.get_train_test_split()

        # 2. Run hyperparameter tuning using cross-validation on the training data.
        study = self._run_hyperparameter_study(splitter)
        best_trial = study.best_trial
        best_params = best_trial.params
        best_params['epochs'] = int(np.ceil(best_trial.user_attrs["best_n_epochs"])) # the optimal number of epochs determined during CV

        logger.info(f"Best trial for Experiment [{experiment_id+1}/{self.args.n_experiments}]: "
                    f"Metric={best_trial.value:.4f}, Params={best_params}")

        # 3. Retrain the final model on the entire training set with the best hyperparameters.
        final_model, final_history, test_loader, processor = self._train_final_model(best_params, splitter)

        # 4. Evaluate the final model on the hold-out test set.
        test_metrics = self._evaluate_final_model(final_model, test_loader, best_trial, processor)

        # 5. Save all results and artifacts for this experiment run.
        handler = ResultHandler(output_dir=self.output_dir, task_type=self.args.task_type,
                                history=final_history, metrics=test_metrics)
        handler.process()
        
        # Save detailed CV records from the HPO phase
        cv_df = pd.DataFrame(self.cv_records)
        cv_path = os.path.join(self.output_dir, "results_CV.csv")
        cv_df.to_csv(cv_path, index=False)
        logger.info(f"Saved detailed CV results for this split to {cv_path}")

        # Save the final test metrics
        scalar_metrics = {k: v for k, v in test_metrics.items() if np.isscalar(v)}
        scalar_metrics['experiment_id'] = experiment_id
        self.test_records.append(scalar_metrics)

        test_df = pd.DataFrame(self.test_records)
        test_path = os.path.join(self.output_dir, "results_test.csv")
        test_df.to_csv(test_path, index=False)

        logger.info(f"\n===== TEST METRICS for Experiment [{experiment_id+1}/{self.args.n_experiments}] =====\n"
                    f"{scalar_metrics}"
                    f"===== =====")

        logger.info(f"\n===== BEST PARAMETERS for Experiment [{experiment_id+1}/{self.args.n_experiments}] =====\n"
                    f"{best_params}\n"
                    f"===== =====")

        logger.info(f"===== Experiment [{self.experiment_id + 1}/{self.args.n_experiments}] COMPLETED. "
                    f"Results saved to {self.output_dir}.")

    def _run_hyperparameter_study(self, splitter: StratifiedBlockSplitter):
        """Sets up and runs an Optuna study for hyperparameter optimization."""
        direction = "maximize" if self.args.task_type == "workhour_classification" else "minimize"
        
        # Pruning
        pruner = MedianPruner(**self.median_pruning)
        
        # build a URI in your results folder
        db_path = os.path.join(self.output_dir, "optuna_study.db")
        storage = f"sqlite:///{db_path}"
        
        study = optuna.create_study(direction=direction,
            storage=storage, study_name=f"experiment_{self.experiment_id}", load_if_exists=True,
            pruner=pruner)
        logger.info(f"Optuna study storage: {storage} (study_name={study.study_name}) with {pruner.__class__.__name__}")
        
        # Making the CV splits
        splitter.get_cv_splits()
        
        objective_func = lambda trial: self._objective(trial, splitter)
        if self.args.optuna_crash_mode == "fail_fast":
            # DEBUG MODE: Let any unhandled exception crash the script for immediate feedback.
            logger.warning("Running in fail_fast mode. Unhandled trial exceptions will crash the experiment.")
            study.optimize(
                objective_func,
                n_trials=self.args.n_optuna_trials
            )
        elif self.args.optuna_crash_mode == "safe":
            # SAFE MODE (DEFAULT): Catch all exceptions, log them, and continue the study.
            logger.info("Running in safe mode. Unhandled trial exceptions will be caught and logged.")
            study.optimize(
                objective_func,
                n_trials=self.args.n_optuna_trials,
                catch=(Exception,)
            )

        return study
    
    def _objective(self, trial: optuna.trial.Trial, splitter: StratifiedBlockSplitter):
        """The objective function for Optuna, performing k-fold cross-validation."""
        trial_args = deepcopy(self.args)

        # --- General Training Hyperparameters ---
        trial_args.lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        trial_args.weight_decay_rate = trial.suggest_float("weight_decay_rate", 1e-5, 1e-2, log=True)
        trial_args.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        trial_args.droprate = trial.suggest_float("droprate", 0.1, 0.6)
        trial_args.enable_bias = trial.suggest_categorical("enable_bias", [True, False])

        # --- STGCN Architecture Hyperparameters ---
        trial_args.graph_conv_type = trial.suggest_categorical("graph_conv_type", ["cheb_graph_conv", "graph_conv"])
        trial_args.act_func = trial.suggest_categorical("act_func", ["glu", "relu", "silu"])
        trial_args.stblock_num = trial.suggest_categorical("stblock_num", [2, 3, 4])
        trial_args.n_his = trial.suggest_categorical("n_his", [12, 18, 24, 30, 36])
        trial_args.Kt = trial.suggest_categorical("Kt", [2, 3])
        trial_args.Ks = trial.suggest_categorical("Ks", [2, 3])
        trial_args.st_main_channels = trial.suggest_categorical("st_main_channels", [32, 64, 96])
        trial_args.st_bottleneck_channels = trial.suggest_categorical("st_bottleneck_channels", [8, 16, 24])
        trial_args.output_channels = trial.suggest_categorical("output_channels", [128, 256, 512])

        # We don't tune epochs directly. We set a max value and let early stopping find the best.
        # This is the max number of epochs the model is allowed to run for in each CV fold.
        trial_args.epochs = self.args.max_epochs

        # --- PRE-FLIGHT CHECK ---
        # Calculate the resulting temporal dimension before building the model.
        ko = trial_args.n_his - (trial_args.stblock_num * 2 * (trial_args.Kt - 1))
        
        if ko <= 0: # This combination is architecturally impossible and will crash.
            logger.warning(
                f"Pruning Trial {trial.number} due to invalid architecture: "
                f"n_his={trial_args.n_his}, Kt={trial_args.Kt}, stblock_num={trial_args.stblock_num} results in Ko={ko}."
            )
            raise optuna.exceptions.TrialPruned()

        # Loop through CV folds.
        fold_metrics, fold_actual_epochs, fold_thresholds = [], [], []
        for fold_num, (train_ids, val_ids) in enumerate(splitter.split()):
            logger.info(f">>> [Trial {trial.number}] Starting CV Fold {fold_num + 1}/{splitter.n_splits} <<<")

            # Load data for the current fold, combine loaders with other necessary data for setup_model
            loaders, processor = self._normalize_data_and_get_dataloaders(trial_args, train_ids, val_ids, test_block_ids=[], splitter=splitter)
            data_for_setup = {**self.input_dict, **loaders}

            epoch_offset = fold_num * trial_args.epochs
            model, criterion, optimizer, scheduler, early_stopping = setup_model(trial_args, data_for_setup)
            _, history = train_model(trial_args, model, criterion, optimizer, scheduler, 
                                    early_stopping=early_stopping, 
                                    train_loader=loaders['train_loader'], 
                                    val_loader=loaders['val_loader'],
                                    trial=trial,
                                    epoch_offset=epoch_offset)
            
            fold_actual_epochs.append(early_stopping.best_epoch)

            if self.args.task_type == "workhour_classification":
                metric_val = history.get('best_val_auc', 0.0)
            else:
                metric_val = min(history.get('val_loss', [np.inf]))
            fold_metrics.append(metric_val)

            # Log the result of this specific CV fold
            cv_record = {
                'experiment_id': self.experiment_id,
                'trial_num': trial.number,
                'fold_num': fold_num,
                'validation_metric': metric_val,
                'epochs_trained': early_stopping.best_epoch,
                **trial.params
            }
            
            if self.args.task_type == "workhour_classification":
                # Reload the best model state to find the threshold that produced it
                model.load_state_dict(early_stopping.best_model_state)
                fold_thresholds.append(find_optimal_threshold(model, loaders['val_loader']))
                cv_record['optimal_threshold'] = fold_thresholds[-1]
                
            self.cv_records.append(cv_record)
        
            # --- Logging: Announce the result of the completed fold ---
            log_msg = f">>> [Trial {trial.number}] Finished CV Fold {fold_num + 1}/{splitter.n_splits} | " \
                    f"Metric: {metric_val:.4f} | Epochs: {early_stopping.best_epoch}"
            if self.args.task_type == "workhour_classification":
                log_msg += f" | Threshold: {fold_thresholds[-1]:.4f}"
            logger.info(log_msg)

        # Aggregate results and save to trial
        avg_metric = np.mean(fold_metrics)
        trial.set_user_attr("best_n_epochs", float(np.mean(fold_actual_epochs)))
        if self.args.task_type == "workhour_classification":
            trial.set_user_attr("optimal_threshold", float(np.mean(fold_thresholds)))

        # --- Logging: Announce the final aggregated result for the trial ---
        logger.info(f"--- Finished Optuna Trial {trial.number} ---")
        logger.info(f"    Average validation metric across {splitter.n_splits} folds: {avg_metric:.4f}")
        logger.info(f"    Average epochs: {np.mean(fold_actual_epochs):.1f}")
        if self.args.task_type == "workhour_classification" and fold_thresholds:
            logger.info(f"    Average optimal threshold: {np.mean(fold_thresholds):.4f}")
        logger.info("-" * 50)

        return avg_metric
    
    def _train_final_model(self, best_params, splitter):
        """Trains the final model on the full training set with best hyperparameters."""
        logger.info("Retraining final model on the full training dataset...")
        
        # Get train and test ids from the splitter
        train_ids = splitter.train_block_ids
        test_ids = splitter.test_block_ids

        final_args = deepcopy(self.args)
        for key, value in best_params.items():
            setattr(final_args, key, value)
        
        loaders, processor = self._normalize_data_and_get_dataloaders(final_args, train_ids, val_block_ids=[], test_block_ids=test_ids, splitter=splitter)
        data_for_setup = {**self.input_dict, **loaders}
        
        model, criterion, optimizer, scheduler, _ = setup_model(final_args, data_for_setup)
        final_model, history = train_model(final_args, model, criterion, optimizer, scheduler, 
                                            early_stopping=None, 
                                            train_loader=loaders['train_loader'], 
                                            val_loader=None)
        return final_model, history, loaders['test_loader'], processor

    def _evaluate_final_model(self, 
                              model: torch.nn.Module, 
                              test_loader: DataLoader, 
                              best_trial: optuna.trial.FrozenTrial,
                              processor: STGCNNormalizer):
        """Evaluates the final, retrained model on the hold-out test set."""
        if self.args.task_type == "workhour_classification":
            # Retrieve the optimal threshold found during CV for the best trial.
            # Use 0.5 as a safe fallback if it's not found.
            optimal_threshold = best_trial.user_attrs.get("optimal_threshold", 0.5)
            logger.info(f"Using optimal classification threshold for evaluation: {optimal_threshold:.4f}")
            
            # Pass this specific threshold to the evaluation function.
            metrics = evaluate_model(self.args, model, test_loader, processor, threshold=optimal_threshold)
        else:
            # For forecasting, no threshold is needed.
            metrics = evaluate_model(self.args, model, test_loader, processor, threshold=None)
        return metrics

    def _save_arguments(self):
        """Saves the experiment configuration arguments to a JSON file."""
        # Define the full path for the arguments file
        args_path = os.path.join(self.output_dir, "args.json")
        
        # Convert the argparse.Namespace object to a dictionary
        args_dict = vars(self.args)
        
        logger.info(f"Saving experiment configuration to {args_path}...")
        try:
            # Open the file and write the dictionary as a JSON object
            # indent=4 makes the file human-readable
            with open(args_path, 'w') as f:
                json.dump(args_dict, f, indent=4)
            logger.info("Successfully saved arguments.")
        except TypeError as e:
            logger.error(f"Could not serialize args to JSON: {e}. Check if args contain non-serializable objects.")
        except Exception as e:
            logger.error(f"An error occurred while saving arguments to {args_path}: {e}")

def main():
    """
    Main entry point to run a single experiment.
    
    This function parses command-line arguments and starts the STGCNExperimentRunner.
    """
    from ....config.args import parse_args
    args = parse_args()
    
    runner = STGCNExperimentRunner(args)
    runner.run_experiment()

if __name__ == '__main__':
    main()