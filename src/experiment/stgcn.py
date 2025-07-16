#!/usr/bin/env python

from copy import deepcopy
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import optuna

from ..preparation.preparer import STGCNDataPreparer
from ..models.STGCN4B.homogeneous.normalizer import STGCNNormalizer
from ..models.STGCN4B.homogeneous.graph_loader import get_data_loaders
from ..models.STGCN4B.homogeneous.train import setup_model, train_model, evaluate_model
from .base import BaseExperimentRunner, TrainingResult

# Set up the main logging
import logging; logger = logging.getLogger(__name__)


class STGCNExperimentRunner(BaseExperimentRunner):
    def __init__(self, args: Any):        
        super().__init__(args)
        self.data_preparer = STGCNDataPreparer(self.args)
    
    def _prepare_data(self) -> Dict[str, Any]:
        logger.info("Handling data preparation for STGCN...")
        data_preparer = STGCNDataPreparer(self.args)
        input_dict = data_preparer.get_input_dict()
        return input_dict
    
    #########################
    # Split preparation
    #########################
    
    def _normalize_split(
            self, args,
            train_block_ids: List[int],
            ) -> Tuple[np.ndarray, np.ndarray, STGCNNormalizer]:
        """Normalizes the feature and target arrays for a given split."""
        # 1. Get train indices and slice arrays to fit the normalizer
        train_indices = self.splitter._get_indices_from_blocks(train_block_ids)
        train_feature_slice = self.input_dict["feature_array"][train_indices]
        train_target_slice = self.input_dict["target_array"][train_indices]
        train_mask_slice = self.input_dict["target_mask"][train_indices] if self.input_dict["target_mask"] is not None else None
        
        # 2. Fit normalizer and transform
        normalizer = STGCNNormalizer()
        
        # Features
        normalizer.fit_features(
            train_array=train_feature_slice,
            feature_names=self.input_dict["feature_names"],
            method=args.normalization_method,
            features_to_skip_norm=args.skip_normalization_for
            )
        norm_feature_array = normalizer.transform_features(full_array=self.input_dict["feature_array"])
        
        # Targets
        if self.args.task_type == "workhour_classification":
            # No need for normalization in classification tasks
            norm_target_array = self.input_dict["target_array"]
        else: # Forecasting tasks
            normalizer.fit_target(
                train_targets=train_target_slice, 
                train_mask=train_mask_slice,
                method='median'
                )
            norm_target_array = normalizer.transform_target(targets=self.input_dict["target_array"])
        
        return norm_feature_array, norm_target_array, normalizer
    
    def _impute_split(self, feature_array, target_array) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-imputes NaN values in the feature and target arrays. In-place operation.
        
        Note: For forecast tasks, we impute to target due to NaN values formed during the target creation.
        This is fine, since we will mask these imputations later. There is no NaN for workhour classification task.
        """
        feature_array[np.isnan(feature_array)] = 0.0
        target_array[np.isnan(target_array)] = 0.0
        return feature_array, target_array
    
    def _load_data_to_tensors(self, **arrays_to_convert) -> Dict[str, torch.Tensor]:
        """Converts a dictionary of numpy arrays to a dictionary of tensors."""
        device = self.input_dict['device']
        return {
            name: torch.from_numpy(arr).float().to(device)
            for name, arr in arrays_to_convert.items()
        }
    
    def _get_split_payload(
            self, args,
            train_block_ids: List[int], val_block_ids: List[int], test_block_ids: List[int]
            )-> Tuple[Dict, STGCNNormalizer]:
        
        # Normalization & Imputation
        norm_feature_array, norm_target_array, normalizer = self._normalize_split(args, train_block_ids)
        feature_array, target_array = self._impute_split(norm_feature_array, norm_target_array)
        
        # Load the processed arrays into tensors
        numpy_payload = {
            "feature": feature_array,
            "target": target_array,
            "target_mask": self.input_dict["target_mask"],
            "target_source": self.input_dict["target_source_array"]
        }
        tensors = self._load_data_to_tensors(**numpy_payload)
        
        # Get DataLoaders
        max_target_offset = 1 if self.args.task_type == "workhour_classification" else max(args.forecast_horizons)
        loaders = get_data_loaders(
            args,
            blocks=self.input_dict["blocks"],
            block_size=self.input_dict["block_size"],
            feature_tensor=tensors["feature"],
            target_tensor=tensors["target"],
            target_mask_tensor=tensors["target_mask"],
            target_source_tensor=tensors["target_source"],
            max_target_offset=max_target_offset,
            train_block_ids=train_block_ids,
            val_block_ids=val_block_ids,
            test_block_ids=test_block_ids
        )
        return loaders, normalizer
    
    #########################
    # HPO
    #########################
    
    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """The objective function for Optuna, performing k-fold cross-validation."""
        trial_args = deepcopy(self.args)
        
        ########## Conditional sampling for n_his ##########
        # Handling the possible invalid architecture
        trial_args.stblock_num = trial.suggest_categorical("stblock_num", [2, 3, 4, 5])
        trial_args.Kt = trial.suggest_categorical("Kt", [2, 3])
        
        # Suggesting n_his with a pre-flight chcke
        all_n_his_options = [12, 18, 24, 30, 36]
        min_required_n_his = trial_args.stblock_num * 2 * (trial_args.Kt - 1) + 1
        valid_n_his_options = [n for n in all_n_his_options if n >= min_required_n_his]
        
        # If no valid options exist for this combination, prune the trial.
        # Although, with the current settings, there should always be valid options.
        if not valid_n_his_options:
            raise optuna.exceptions.TrialPruned(
                f"No valid n_his for stblock_num={trial_args.stblock_num} and Kt={trial_args.Kt}"
            )
        trial_args.n_his = trial.suggest_categorical("n_his", valid_n_his_options)
        
        ########## End of Conditional sampling for n_his ##########
        
        # --- General Training Hyperparameters ---
        trial_args.lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        trial_args.weight_decay_rate = trial.suggest_float("weight_decay_rate", 1e-5, 1e-2, log=True)
        trial_args.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        trial_args.droprate = trial.suggest_float("droprate", 0.1, 0.6)
        trial_args.enable_bias = trial.suggest_categorical("enable_bias", [True, False])
        
        # --- STGCN Architecture Hyperparameters ---
        trial_args.graph_conv_type = trial.suggest_categorical("graph_conv_type", ["gcn", "cheb"])
        trial_args.act_func = trial.suggest_categorical("act_func", ["glu", "relu", "silu"])
        trial_args.Ks = trial.suggest_categorical("Ks", [2, 3])
        trial_args.st_main_channels = trial.suggest_categorical("st_main_channels", [32, 64, 96])
        trial_args.st_bottleneck_channels = trial.suggest_categorical("st_bottleneck_channels", [8, 16, 24])
        trial_args.output_channels = trial.suggest_categorical("output_channels", [128, 256, 512])
        
        # We don't tune epochs directly. We set a max value and let early stopping find the best.
        # This is the max number of epochs the model is allowed to run for in each CV fold.
        trial_args.epochs = self.args.epochs
        
        return trial_args
    
    ##########################
    # Experiment execution
    ##########################
    
    def _train_one_fold(
        self,
        trial:              optuna.trial.Trial,
        trial_params:       Dict[str, Any],
        fold_index:         int = None,
        *,
        train_block_ids:    List[int],
        val_block_ids:      List[int],
        ) -> TrainingResult:
        """Trains and evaluates a single fold in cross-validation for HPO."""
        
        # NOTE: Pruning callback is handled inside train_model()
        
        # 2. Get split payload
        loaders, normalizer = self._get_split_payload(
            trial_params, 
            train_block_ids, val_block_ids, test_block_ids=[])
        
        # 3. Training
        data_for_setup = {**self.input_dict, **loaders}
        model, criterion, optimizer, scheduler = setup_model(
            trial_params, 
            data_for_setup)
        
        model, history, training_result = train_model(
            trial_params, model, criterion, optimizer, scheduler, 
            train_loader    = loaders['train_loader'], 
            val_loader      = loaders['val_loader'],
            trial           = trial,
            epoch_offset    = fold_index * trial_params.epochs)
                
        return training_result
    
    ##########################
    # Final Training & Evaluation
    ##########################

    def _train_and_evaluate_final_model(
        self, 
        params:             Dict[str, Any],
        epochs:             int = None,
        threshold:          float = None,
        *,
        train_block_ids:    List[int],
        val_block_ids:      List[int], # For run_mode="test", we do have a validation set
        test_block_ids:     List[int],
        ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """Trains and evalutes the final model."""
        
        # Get the final parameters
        final_params = deepcopy(self.args)
        for key, value in vars(params).items():
            setattr(final_params, key, value)

        # Expose the epochs to args
        # NOTE. Using a heuristic: train for slightly longer on the full dataset.
        if epochs is not None:
            final_params.epochs = int(np.ceil(epochs * 1.1))
            logger.info(f"Inferred optimal epochs from CV: {epochs:.0f}. "
                        f"Training final model for {final_params.epochs} rounds (1.1x).")
        
        # Get split payload
        loaders, normalizer = self._get_split_payload(
            final_params, 
            train_block_ids, val_block_ids=val_block_ids, test_block_ids=test_block_ids)
        
        # Training
        data_for_setup = {**self.input_dict, **loaders}
        model, criterion, optimizer, scheduler = setup_model(
            final_params, 
            data_for_setup)
        
        model, history, training_result = train_model(
            final_params, model, criterion, optimizer, scheduler, 
            train_loader    = loaders['train_loader'], 
            val_loader      = loaders['val_loader'] if val_block_ids else None,
            trial           = None,
            epoch_offset    = 0
        )
        
        # Determining the threshold if not already given (from HPO)
        if self.args.task_type == "workhour_classification":
            if threshold is None:
                threshold = training_result.optimal_threshold
            logger.info(f"Using optimal classification threshold for evaluation: {threshold:.4f}")
        
        # Evaluation
        metrics, model_outputs = evaluate_model(
            args        = self.args, 
            model       = model, 
            test_loader = loaders['test_loader'], 
            normalizer  = normalizer, 
            threshold   = threshold
        )
        
        return model, history, metrics, model_outputs
    
def main():
    """
    Main entry point to run a single experiment.
    
    This function parses command-line arguments and starts the STGCNExperimentRunner.
    """
    from ..config.args import parse_args
    args = parse_args()
    
    runner = STGCNExperimentRunner(args)
    runner.run()

if __name__ == '__main__':
    main()