import os
import sys
import json
import pathlib
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from argparse import Namespace
from copy import deepcopy
import optuna
from optuna.pruners import MedianPruner

from ..preparation.split import StratifiedBlockSplitter
from ..utils.result_handler import ResultHandler
from ..utils.tracking import TrainingResult, TrainingHistory

import logging; logger = logging.getLogger(__name__)


class BaseExperimentRunner(ABC):
    """
    An abstract base class for running machine learning experiments.
    
    This class provides a common structure for setting up experiments,
    handling arguments, and saving results. Model-specific runners should
    inherit from this class and implement the abstract methods.
    """
    def __init__(self, args: any):
        """
        Initializes the BaseExperimentRunner.

        The full experiment is designed to be run in parallel for different random seeds.
        
        Args:
            args: A configuration object (e.g., from argparse) containing all
                  necessary parameters. 
                  
                  If runing a full experiment, it MUST include:
                    - `split_id`: A unique integer for this experiment run, used for seeding.
                    - `seed`: A base random seed.
                    - `n_optuna_trials`: The number of HPO trials to run.
        """
        self.setup_logging(level=args.log_level)
        
        # Args
        self.args = args
        # A unique ID for this specific experiment run (e.g., from SLURM_ARRAY_TASK_ID)
        self.experiment_id = args.experiment_id # (default is args.experiment_id==0)
        # Create a unique seed for this run to ensure data splits are different
        self.seed = args.seed + self.experiment_id

        # Set up the specific output directory for this run's artifacts
        self.output_dir = args.output_dir
        logger.info(f"Experiment outputs will be saved in: {self.output_dir}")

        # Save the experiment configuration arguments to a JSON file
        self._save_arguments()

        # Set run mode
        self.run_mode = args.run_mode

        # Initialize records
        self.cv_records: List[Dict[str, Any]] = []
        self.test_records: List[Dict[str, Any]] = []

    @staticmethod
    def setup_logging(level: str='INFO'):
        """
        Sets up the root logger for the experiment.
        This is a static method so it can be called without an instance.
        """
        # Convert string level to logging constant (e.g., 'INFO' -> logging.INFO)
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {level}')
        
        logging.basicConfig(
            level    = numeric_level,
            format   = '%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s',
            handlers = [logging.StreamHandler(sys.stdout)]
        )
        # Suppress Optuna's INFO messages to keep the logs cleaner
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info(f"Logging configured at level: {level.upper()}")

    #########################
    # Main Orchestrators
    #########################
    
    def run(self) -> None:
        """Main entry point to run the experiment."""
        # Prepare the input dictionary
        self.input_dict = self._prepare_data()
        
        # Initialize the stratified splitter, and get train/test splits
        self.splitter = StratifiedBlockSplitter(
            output_dir=self.output_dir, 
            blocks=self.input_dict['blocks'], 
            stratum_size=self.args.stratum_size, 
            seed=self.seed
        )
        self.splitter.get_train_test_split()

        if self.run_mode == 'single_run':
            logger.info(f"===== Starting Single Run | Seed: {self.seed} =====")
            self._run_test()
        elif self.run_mode == 'experiment':
            logger.info(f"===== Starting HPO Experiment [{self.experiment_id+1}] | Seed: {self.seed} =====")
            self._run_experiment()
        else:
            raise ValueError(f"Invalid run_mode: {self.run_mode}. Choose from 'experiment' or 'single_run'.")
    
    def _run_test(self) -> None:
        """Single run without CV or HPO."""
        
        # Train and evaluate the final model using the best hyperparameters
        train_block_ids, val_block_ids = self.splitter.get_single_split()
        test_block_ids = self.splitter.test_block_ids
        
        # Setup final model
        logger.info("Setting up the single-run model...")
        model = self._setup_model(args=self.args)
        
        # Train and evaluate the final model
        trained_model, history, metrics, model_outputs = self._train_and_evaluate_final_model(
            model               = model,
            final_params        = self.args, 
            epochs              = None,
            threshold           = None,
            train_block_ids     = train_block_ids, 
            val_block_ids       = val_block_ids,
            test_block_ids      = test_block_ids
        )
        
        # Save the results
        self._save_results(trained_model, history, metrics, model_outputs)
    
    def _run_experiment(self) -> None:
        """Executes the full pipeline for a single experiment instance."""
        # Run the hyperparameter optimization study and get the best trial
        study = self._run_hyperparameter_study()

        # Get the best trial parameters and attributes
        best_trial  = study.best_trial
        best_params_dict = best_trial.params
        the_best_epoch = int(best_trial.user_attrs.get("best_n_epochs", 0))
        optimal_thr = best_trial.user_attrs.get("optimal_threshold", 0.5)
        
        # Save the best hyperparameters.
        self._save_dict(best_params_dict, "best_hyperparameters.json")
        
        # Create a full configuration object for the final model
        final_params = deepcopy(self.args)
        for key, value in best_params_dict.items():
            setattr(final_params, key, value)
        
        # Train and evaluate the final model using the best hyperparameters

        ## Get train (and optionally validation)
        if getattr(self.args, "use_validation_in_final", False):
            train_block_ids, val_block_ids = self.splitter.get_final_train_valid_split()
            epochs_for_final=None
        else:
            train_block_ids = self.splitter.train_block_ids
            val_block_ids = []
            epochs_for_final=the_best_epoch
        ## Get test
        test_block_ids = self.splitter.test_block_ids
        
        # Setup final model
        logger.info("Setting up the final model with the best hyperparameters...")
        model = self._setup_model(args=final_params)
        
        # Train and evaluate the final model
        trained_model, history, metrics, model_outputs = self._train_and_evaluate_final_model(
            model               = model,
            final_params        = final_params, 
            epochs              = epochs_for_final,
            threshold           = optimal_thr,
            train_block_ids     = train_block_ids, 
            val_block_ids       = val_block_ids,
            test_block_ids      = test_block_ids
        )

        # Save the results
        self._save_results(trained_model, history, metrics, model_outputs, study)
    
    #########################
    # Abstract Methods
    #########################
    
    # Data preparation
    @abstractmethod
    def _prepare_data(self) -> Dict[str, Any]:
        """
        Loads and prepares data, returns a dictionary of necessary data objects.
        """
        pass
    
    @abstractmethod
    def _normalize_split(self):
        """Abstract method for normalizing the split data."""
        pass
    
    @abstractmethod
    def _impute_split(self):
        """Abstract method for imputing missing values in the split data."""
        pass
    
    @abstractmethod
    def _get_split_payload(self):
        """Abstract method to get the split data payload for training."""
        pass
    
    # HPO-related methods
    @abstractmethod
    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Optuna trials based on the defined search space."""
        pass
    
    @abstractmethod
    def _setup_model(self, args: Any) -> Any:
        """Sets up the model for training and evaluation."""
        pass
    
    @abstractmethod
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
        """
        Trains and evaluates a single fold in cross-validation for HPO.

        Steps:
        1. Sets up pruning callback.
        2. Get split payload
        3. Trains model.
        4. Gets TrainingResult(metric, best_epoch, threshold).
        """
        pass
    
    @abstractmethod
    def _cleanup_after_trial(self) -> None:
        """
        Cleans up resources after a trial is completed.
        This is called at the end of each trial in HPO.
        """
        pass

    @abstractmethod
    def _cleanup_after_fold(self) -> None:
        """
        Cleans up resources after a fold is completed.
        This is called at the end of each fold in cross-validation.
        """
        pass
    
    # General training and evaluation methods
    @abstractmethod
    def _train_and_evaluate_final_model(
        self, 
        model:              Any,
        final_params:       Namespace,
        epochs:             int = None,
        threshold:          float = None,
        *,
        train_block_ids:    List[int],
        val_block_ids:      List[int], # For run_mode="single_run", we do have a validation set
        test_block_ids:     List[int],
        ) -> Tuple[Any, TrainingHistory, Dict[str, Any], Dict[str, Any]]:
        """Trains the final model."""
        pass
        
    #########################
    # HPO Helpers
    #########################
    
    def _run_hyperparameter_study(self) -> optuna.Study:
        """Sets up and runs an Optuna study."""
        # Storage
        db_path = os.path.join(self.output_dir, "optuna_study.db")
        storage_url = f"sqlite:///{db_path}"
        
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 20}}  # Wait 20 seconds if DB is locked
        )
        logger.info(f"Set Optuna study storage to: {storage_url} with a 20s timeout.")
                            
        # Make the cross-validation splits
        self.splitter.get_cv_splits()
        
        # Pruner configuration
        median_pruner_args = {
            "n_startup_trials": self.args.n_startup_trials,
            "n_warmup_steps": self.args.n_warmup_steps,
            "interval_steps": self.args.interval_steps
        }
        pruner = MedianPruner(**median_pruner_args)
                
        # Set the optimization direction based on task type
        direction = "maximize" if self.args.task_type == "workhour_classification" else "minimize"
        
        # Create (or load) the study
        study = optuna.create_study(
            direction=direction,
            storage=storage, 
            study_name=f"experiment_{self.experiment_id}", 
            load_if_exists=True,
            pruner=pruner
        )
        
        # Objective function for Optuna trials
        objective_func = lambda trial: self._objective(trial)
        
        # Run the study with the specified number of trials and jobs
        if self.args.optuna_crash_mode == "fail_fast":
            logger.warning("Running in fail_fast mode. Unhandled trial exceptions will crash the experiment.")
            study.optimize(
                objective_func,
                n_trials=self.args.n_optuna_trials,
                n_jobs=self.args.n_jobs_in_hpo,
            )
        elif self.args.optuna_crash_mode == "safe":
            logger.info("Running in safe mode. Unhandled trial exceptions will be caught and logged.")
            study.optimize(
                objective_func,
                n_trials=self.args.n_optuna_trials,
                n_jobs=self.args.n_jobs_in_hpo,
                catch=(Exception,)
            )
        
        return study
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna trials.

        Note:
        - Subclasses should handle the pruning themselves.        
        """
        logger.info("=" * 50)
        logger.info(f">>> Starting trial {trial.number} <<<")
        
        try:
            trial_args = self._suggest_hyperparams(trial)
            model = self._setup_model(args=trial_args)
            
            fold_results: List[TrainingResult] = []
            for fold_num, (train_idx, val_idx) in enumerate(self.splitter.split()):
                logger.info(f">>> Starting CV Fold {fold_num + 1}/{self.splitter.n_splits} <<<")
                
                result = self._train_one_fold(
                    model           = model,
                    trial           = trial,
                    trial_params    = trial_args,
                    fold_index      = fold_num,
                    train_block_ids = train_idx,
                    val_block_ids   = val_idx
                )
                fold_results.append(result)

                # Light cleanup between folds
                self._cleanup_after_fold()
                
                # Keep detailed record
                self.cv_records.append({
                    "experiment_id":        self.experiment_id,
                    "trial":                trial.number,
                    "fold":                 fold_num,
                    "metric":               result.metric,
                    "best_epoch":           result.best_epoch,
                    "optimal_threshold":    result.optimal_threshold,
                    **trial.params
                })
                
                # Logging: Announce the result of the completed fold
                logger.info(f">>> Finished [Trial {trial.number}] CV Fold {fold_num + 1}/{self.splitter.n_splits} <<<")
                log_msg = f"Metric: {result.metric:.4f} | Epochs: {result.best_epoch}"
                if self.args.task_type == "workhour_classification":
                    log_msg += f" | Optimal threshold: {result.optimal_threshold:.4f}"
                logger.info(log_msg)
            
            # Average metric
            avg_metric = float(np.mean([r.metric for r in fold_results]))
            
            # Best number of epochs
            epochs = [r.best_epoch for r in fold_results if r.best_epoch is not None]
            best_n_epochs = float(np.mean(epochs)) if epochs else None
            trial.set_user_attr("best_n_epochs", best_n_epochs)
            
            # Optimal threshold
            best_thresh = None
            if self.args.task_type == "workhour_classification":
                thresholds = [r.optimal_threshold for r in fold_results
                            if r.optimal_threshold is not None]
                best_thresh = float(np.mean(thresholds)) if thresholds else None
                trial.set_user_attr("optimal_threshold", best_thresh)
            
            # Logging: Announce the final aggregated result for the trial
            logger.info(f">>> Finished Optuna Trial {trial.number} <<<")
            logger.info(" Average validation metric across %d folds: %.4f",
                        self.splitter.n_splits, avg_metric)
            logger.info(" Average epochs: %s",
                        f"{best_n_epochs:.1f}" if best_n_epochs is not None else "n/a")
            if self.args.task_type == "workhour_classification":
                logger.info(" Average optimal threshold: %s",
                            f"{best_thresh:.4f}" if best_thresh is not None else "n/a")
            logger.info("=" * 50)

            return avg_metric
        
        finally:
            # Dropping references so empty_cache can actually release
            # NOTE: None assignment is a bit more robust than 'del'.
            model = None
            trial_args = None
            fold_results = None
            # Cleanup resources after the trial
            self._cleanup_after_trial()
    
    #########################
    # Saves
    #########################
    
    def _save_dict(self, obj: dict, filename: str) -> pathlib.Path:
        """
        Dump *obj* to `output_dir/filename` in both JSON (human-readable) and
        CSV (easy for spreadsheets). Returns the JSON file path.
        """
        out_dir = pathlib.Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / filename
        csv_path  = json_path.with_suffix(".csv")

        with json_path.open("w") as f:
            json.dump(obj, f, indent=4)

        # flat table → one row
        pd.json_normalize(obj).to_csv(csv_path, index=False)

        logger.info("Saved best hyper-parameters to %s and %s", json_path, csv_path)
        return json_path

    def _save_arguments(self):
        """Saves the experiment configuration arguments to a JSON file."""
        args_path = os.path.join(self.output_dir, "args.json")
        args_dict = vars(self.args)
        
        logger.info(f"Saving experiment configuration to {args_path}...")
        try:
            with open(args_path, 'w') as f:
                json.dump(args_dict, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save arguments: {e}", exc_info=True)
    
    def _save_results(
        self, 
        model: Any, 
        history: TrainingHistory, 
        metrics: Dict[str, Any],
        model_outputs: Dict[str, Any],
        study: optuna.Study = None
        ) -> None:
        """Handles saving all experiment artifacts."""
        # Save CV records (if it's an experiment run)
        if self.run_mode == 'experiment':
            cv_path = os.path.join(self.output_dir, "results_CV.csv")
            cv_results_df = pd.DataFrame(self.cv_records)
            cv_results_df.to_csv(cv_path, index=False)
            logger.info(f"Saved detailed CV results for this split to {cv_path}")
        
        # Save final test metrics
        scalar_metrics = {k: v for k, v in metrics.items() if np.isscalar(v)}
        if self.run_mode == 'experiment':
            scalar_metrics['experiment_id'] = self.experiment_id
        
        # Save test metrics
        test_path = os.path.join(self.output_dir, "results_test.csv")
        test_results_df = pd.DataFrame([scalar_metrics])
        test_results_df.to_csv(test_path, index=False)
        logger.info(f"Test metrics saved to {test_path}")
        
        handler = ResultHandler(
            output_dir=self.output_dir, 
            task_type=self.args.task_type,
            history=history, 
            metrics=metrics, 
            model_outputs=model_outputs,
            model=model
        )
        handler.process()