#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import MedianPruner
from copy import deepcopy
from typing import Dict, Any, List, Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import plotly

from ....utils.filename_util import get_data_filename
from ....utils.block_split import StratifiedBlockSplitter
from ....utils.train_utils import ResultHandler
from .graph_loader import load_data
from .processor import NumpyDataProcessor
from .train import setup_model, train_model, evaluate_model, find_optimal_threshold

# Set up logging
import logging, sys
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
            output: 
        """
        self.args = args
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

        # Output directories for this specific experiment
        self.main_output_dir = args.output_dir
        logger.info(f"Experiment outputs will be saved in: {self.main_output_dir}")
        self.splits_dir = os.path.join(self.main_output_dir, "splits")
        os.makedirs(self.splits_dir, exist_ok=True)
        self.final_results_dir = os.path.join(self.main_output_dir, "results")
        os.makedirs(self.final_results_dir, exist_ok=True)

    def load_data_file(self):
        """
        Loads the pre-computed NumPy arrays and other metadata.
        This is the one-time disk I/O.
        """
        args = self.args

        # Deriving file name from arguments
        fname = get_data_filename()
        self.data_filename = fname
        path = os.path.join(args.data_dir, "processed", fname)
        
        # Loading the file
        logger.info(f"Loading pre-computed NumPy input from {path}")
        numpy_input = torch.load(path)

        # Determine device
        device = torch.device("cuda" if args.enable_cuda and torch.cuda.is_available() else "cpu")

        # Convert graph structures to tensors centrally upon loading
        adj_matrix_tensor = torch.from_numpy(numpy_input["adjacency_matrix"]).float().to(device)
        masked_adj_dict_tensor = {
            k: torch.from_numpy(v).float().to(device)
            for k, v in numpy_input["masked_adjacencies"].items()
            }

        # Creatign the input dict
        self.input_dict = {
            "device": device,
            
            # Data indices in block format
            "blocks": numpy_input["blocks"],

            # Main data as NumPy arrays
            "feature_array": numpy_input["feature_array"], # Shape (T, R, F)
            "feature_names": numpy_input["feature_names"],
            "n_features": numpy_input["n_features"],

            # Graph structure
            "room_uris": numpy_input["room_uris"],
            "n_nodes": len(numpy_input["room_uris"]),

            # Adjacency
            "adjacency_matrix": adj_matrix_tensor,
            "masked_adjacencies": masked_adj_dict_tensor,

            # Masks
            "target_mask": numpy_input.get("target_mask", None)
        }
        if args.task_type == "workhour_classification":
            self.input_dict["targets"] = numpy_input["workhour_labels"]
        elif args.task_type == "consumption_forecast":
            self.input_dict["targets"] = numpy_input["consumption_values"]
        elif args.task_type == "measurement_forecast":
            target_name = f"{self.measurement_variable}_values"
            self.input_dict["targets"] = numpy_input[target_name]
        else:
            raise ValueError(f"Unknown task type: {args.task_type}")

        logger.info("NumPy data loaded and ready for processing.")

    def _process_and_load_data(self, args: Any,
                               train_block_ids: List[int], val_block_ids: List[int], test_block_ids: List[int],
                               splitter: StratifiedBlockSplitter) -> Tuple[Dict, NumpyDataProcessor]:
        """
        Handles all data processing for a given fold using the fast NumPy workflow.
        Returns the final data loaders and the fitted processor.
        """
        device = self.input_dict['device']
        
        # --- 1. Get train indices and slice arrays to fit the processor ---
        train_indices = splitter._get_indices_from_blocks(train_block_ids)
        train_feature_slice = self.input_dict["feature_array"][train_indices]
        train_target_slice = self.input_dict["targets"][train_indices]
        train_mask_slice = self.input_dict["target_mask"][train_indices] if self.input_dict["target_mask"] is not None else None

        # --- 2. Fit processor, transform full arrays, and impute ---
        processor = NumpyDataProcessor()

        ##### X #####
        processor.fit_features(train_feature_slice)
        norm_feature_array = processor.transform_features(self.input_dict["feature_array"])
        norm_feature_array[np.isnan(norm_feature_array)] = 0.0
        # Convert final NumPy array to the required dictionary format
        T = norm_feature_array.shape[0]
        feature_matrices_dict = {t: norm_feature_array[t] for t in range(T)}
        # Move all feature matrices to device
        for t in feature_matrices_dict:
            feature_matrices_dict[t] = torch.from_numpy(feature_matrices_dict[t]).float().to(device)

        ##### y #####
        if self.args.task_type == "workhour_classification":
            targets_numpy = self.input_dict["targets"]
        else: # Forecasting tasks
            processor.fit_target(train_target_slice, train_mask_slice)
            targets_numpy = processor.transform_target(self.input_dict["targets"])

        # For measurement forecast task, we impute to target
        # This is fine, since we will mask these imputations later
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
        
        # --- 5. Get DataLoaders ---
        loaders = load_data(
            args,
            blocks=self.input_dict["blocks"],
            feature_matrices=feature_matrices_dict,
            targets=final_targets,
            target_mask=final_mask,
            train_block_ids=train_block_ids,
            val_block_ids=val_block_ids,
            test_block_ids=test_block_ids
        )
        return loaders, processor

    def run_experiment(self):
        """Executes the entire experimental pipeline."""
        logger.info("Starting experiment orchestration...")
        all_blocks = self.input_dict["blocks"]

        ########## Outer Loop: Multiple Train-Test Splits ##########
        for i in range(self.args.n_outer_splits):
            seed = self.args.seed + i
            logger.info(f"--- Starting Outer Loop Iteration {i+1}/{self.args.n_outer_splits} (Seed: {seed}) ---")

            # 1. Create a new splitter with a new seed for a fresh train-test split.
            splitter = StratifiedBlockSplitter(output_dir=self.splits_dir, blocks=all_blocks, stratum_size=self.args.stratum_size, seed=seed)
            splitter.get_train_test_split()

            # 2. Run the inner loop: hyperparameter tuning with CV.
            study = self._run_hyperparameter_study(splitter, outer_split_num=i)
            best_trial = study.best_trial
            best_params = best_trial.params
            best_params['epochs'] = int(np.ceil(best_trial.user_attrs["best_n_epochs"])) # the optimal number of epochs determined during CV

            logger.info(f"Best trial for outer loop {i+1}: Metric={best_trial.value:.4f}, Params={best_params}")

            # 3. Retrain the final model on the entire training set with the best hyperparameters.
            final_model, final_history, test_loader, processor = self._train_final_model(best_params, splitter)
            test_metrics = self._evaluate_final_model(final_model, test_loader, best_trial, processor)

            ### Saving ###
            iter_results_dir = os.path.join(self.final_results_dir, f"split_{i}")
            handler = ResultHandler(output_dir=iter_results_dir, task_type=self.args.task_type,
                                    history=final_history, metrics=test_metrics)
            handler.process()
            ### Saving ###

            # Store test metrics
            scalar_metrics = {k: v for k, v in test_metrics.items() if np.isscalar(v)}
            scalar_metrics['outer_split'] = i
            self.test_records.append(scalar_metrics)
            logger.info(f"Test metrics for outer loop {i+1}: {scalar_metrics}")

        # 5. Report and save final results across all outer loop iterations
        self._report_and_save_final_results()

    def _run_hyperparameter_study(self, splitter: StratifiedBlockSplitter, outer_split_num: int):
        """Sets up and runs an Optuna study for hyperparameter optimization."""
        direction = "maximize" if self.args.task_type == "workhour_classification" else "minimize"
        
        # Pruning
        pruner = MedianPruner(**self.median_pruning)
        
        # build a URI in your results folder
        db_path = os.path.join(self.main_output_dir, "optuna_study.db")
        storage = f"sqlite:///{db_path}"
        
        study = optuna.create_study(direction=direction,
            storage=storage, study_name=f"outer_loop_{outer_split_num}", load_if_exists=True,
            pruner=pruner)
        logger.info(f"Optuna study storage: {storage} (study_name={study.study_name}) with {pruner.__class__.__name__}")

        objective_func = lambda trial: self._objective(trial, splitter, outer_split_num)
        if self.args.optuna_crash_mode == "fail_fast":
            # DEBUG MODE: Let any unhandled exception crash the script for immediate feedback.
            logger.warning("Running in fail_fast mode. Unhandled trial exceptions will crash the experiment.")
            study.optimize(
                objective_func, 
                n_trials=self.args.n_optuna_trials,
                n_jobs=self.args.n_jobs
            )
        elif self.args.optuna_crash_mode == "safe":
            # SAFE MODE (DEFAULT): Catch all exceptions, log them, and continue the study.
            logger.info("Running in safe mode. Unhandled trial exceptions will be caught and logged.")
            study.optimize(
                objective_func, 
                n_trials=self.args.n_optuna_trials,
                n_jobs=self.args.n_jobs,
                catch=(Exception,)
            )

        return study
    
    def _objective(self, trial: optuna.trial.Trial, splitter: StratifiedBlockSplitter, outer_split_num: int):
        """The objective function for Optuna, performing k-fold cross-validation."""
        splitter.get_cv_splits()
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

        # We don't tune epochs directly. We set a max value and let early stopping find the best.
        # This is the max number of epochs the model is allowed to run for in each CV fold.
        trial_args.epochs = 300

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
            loaders, processor = self._process_and_load_data(trial_args, train_ids, val_ids, test_block_ids=[], splitter=splitter)
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
                metric_val = max(history['val_metrics'].get('f1', [0]))
            else:
                metric_val = min(history.get('val_loss', [np.inf]))
            fold_metrics.append(metric_val)

            # Log the result of this specific CV fold
            cv_record = {
                'outer_split': outer_split_num,
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
        """Trains the final model on the full training set."""
        logger.info("Retraining final model on all available training data...")
        
        # Get train and test ids from the splitter
        train_ids = splitter.train_block_ids
        test_ids = splitter.test_block_ids

        final_args = deepcopy(self.args)
        for key, value in best_params.items():
            setattr(final_args, key, value)
        
        loaders, processor = self._process_and_load_data(final_args, train_ids, val_block_ids=[], test_block_ids=test_ids, splitter=splitter)
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
                              processor: NumpyDataProcessor):
        """Evaluates the final model on the hold-out test set."""
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

    def _report_and_save_final_results(self):
        """Aggregates, logs, and saves all collected results to CSV files."""
        logger.info("---" * 20 + "\n--- Aggregating and Saving Final Results ---\n" + "---" * 20)
        
        # 1. Save detailed CV records
        cv_df = pd.DataFrame(self.cv_records)
        cv_path = os.path.join(self.final_results_dir, "cv_results_per_fold.csv")
        cv_df.to_csv(cv_path, index=False)
        logger.info(f"Saved detailed CV results to {cv_path}")

        # 2. Save detailed test records
        if not self.test_records:
            logger.warning("No test records to save.")
            return
            
        test_df = pd.DataFrame(self.test_records)
        test_path = os.path.join(self.final_results_dir, "test_results_per_split.csv")
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved test results per split to {test_path}")

        # 3. Calculate and save the final summary report
        summary_df = test_df.describe().loc[['mean', 'std']]
        summary_path = os.path.join(self.final_results_dir, "final_summary_metrics.csv")
        summary_df.to_csv(summary_path)
        
        logger.info("\n--- Aggregated Metrics (Mean ± Std) ---\n" + summary_df.to_string(float_format="%.4f") + "\n" + "---" * 20)

        # 4. Save the name of the data file used for this experiment
        data_source_path = os.path.join(self.final_results_dir, "data_source.txt")
        with open(data_source_path, 'w') as f:
            f.write(f"Experiment data was sourced from the following file:\n")
            f.write(f"{self.data_filename}\n")
        logger.info(f"Saved data source filename to {data_source_path}")

        # 5. Generate and save Optuna analysis visualizations
        if not cv_df.empty:
            self._generate_optuna_visualizations(cv_df)
        else:
            logger.warning("CV dataframe is empty, skipping Optuna visualizations.")

    def _generate_optuna_visualizations(self, cv_df: pd.DataFrame):
        """
        Generates and saves plots related to the Optuna hyperparameter study.

        This method produces:
        - A violin plot of the validation metric distribution (validation_metric_distribution.png).
        - Interactive HTML plots from Optuna for the first outer loop's study:
        - Optimization history (optuna_optimization_history.html).
        - Hyperparameter importances (optuna_param_importances.html).
        - Slice plot (optuna_slice_plot.html).
        """
        logger.info("--- Generating Optuna Analysis Visualizations ---")
        optuna_analysis_dir = os.path.join(self.final_results_dir, "optuna_analysis")
        os.makedirs(optuna_analysis_dir, exist_ok=True)
        logger.info(f"Saving Optuna plots to: {optuna_analysis_dir}")

        # 1. Plot the distribution of validation metrics across all folds and trials
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=cv_df, x='validation_metric', inner='quartile', color='lightblue')
        sns.stripplot(data=cv_df, x='validation_metric', color='darkblue', alpha=0.4, jitter=0.1)
        metric_name = "F1 Score" if self.args.task_type == "workhour_classification" else "Loss (MSE)"
        plt.title(f'Distribution of Validation {metric_name} Across All CV Folds & Trials', fontsize=16)
        plt.xlabel(f'Validation {metric_name}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        dist_path = os.path.join(optuna_analysis_dir, 'validation_metric_distribution.png')
        plt.savefig(dist_path, dpi=300)
        plt.close()
        logger.info(f"Saved validation metric distribution plot to {dist_path}")

        # 2. Generate and save standard Optuna plots from the first outer loop's study
        logger.info("Loading study 'outer_loop_0' to generate standard Optuna plots...")
        try:
            db_path = os.path.join(self.main_output_dir, "optuna_study.db")
            storage = f"sqlite:///{db_path}"
            study_to_plot = optuna.load_study(study_name="outer_loop_0", storage=storage)
            
            # Plot Optimization History
            history_fig = optuna.visualization.plot_optimization_history(study_to_plot)
            history_path = os.path.join(optuna_analysis_dir, 'optuna_optimization_history.html')
            history_fig.write_html(history_path)
            logger.info(f"Saved optimization history plot to {history_path}")

            # Plot Hyperparameter Importances
            if len(study_to_plot.trials) > 1:
                importance_fig = optuna.visualization.plot_param_importances(study_to_plot)
                importance_path = os.path.join(optuna_analysis_dir, 'optuna_param_importances.html')
                importance_fig.write_html(importance_path)
                logger.info(f"Saved parameter importances plot to {importance_path}")
            else:
                logger.warning("Skipping parameter importance plot: not enough trials.")

            # Plot Slice
            slice_fig = optuna.visualization.plot_slice(study_to_plot)
            slice_path = os.path.join(optuna_analysis_dir, 'optuna_slice_plot.html')
            slice_fig.write_html(slice_path)
            logger.info(f"Saved slice plot to {slice_path}")

        except Exception as e:
            logger.error(f"Could not generate Optuna plots. Error: {e}")

def main():
    """
    The main entry point for the script.
    - Parses arguments.
    - Initializes the ExperimentRunner.
    - Starts the experiment.
    """
    from ....config.args import parse_args
    args = parse_args()
    
    runner = ExperimentRunner(args)
    runner.run_experiment()

if __name__ == '__main__':
    main()
