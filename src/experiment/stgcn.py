#!/usr/bin/env python

from copy import deepcopy
from typing import Dict, Any, List, Tuple, Union
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import optuna
import gc
import torch._dynamo
import torch._inductor.codecache
import torch._inductor.utils

from ..preparation.preparer import STGCNDataPreparer
from ..models.STGCN4B.homogeneous.normalizer import STGCNNormalizer
from ..models.STGCN4B.homogeneous.graph_loader import get_data_loaders
from ..models.STGCN4B.homogeneous.train import train_model, evaluate_model
from ..models.STGCN4B.homogeneous.setup import create_gso, create_optimizer, create_scheduler
from ..models.STGCN4B.homogeneous.models import HomogeneousSTGCN
from ..utils.tracking import TrainingResult, TrainingHistory
from .base import BaseExperimentRunner

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
        return {
            name: torch.from_numpy(arr).float()
            for name, arr in arrays_to_convert.items()
        }
    
    def _get_split_payload(
            self, 
            args: Any,
            seed: int,
            train_block_ids: List[int], 
            val_block_ids: List[int], 
            test_block_ids: List[int]
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
        max_horizon = 0 if self.args.task_type == "workhour_classification" else max(args.forecast_horizons)
        loaders = get_data_loaders(
            args=args,
            seed=seed,
            blocks=self.input_dict["blocks"],
            block_size=self.input_dict["block_size"],
            feature_tensor=tensors["feature"],
            target_tensor=tensors["target"],
            target_mask_tensor=tensors["target_mask"],
            target_source_tensor=tensors["target_source"],
            max_horizon=max_horizon,
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
        
        ########## Preflight pruning ##########
        # Handling the possible invalid architecture
        trial_args.stblock_num = trial.suggest_int("stblock_num", 2, 5)
        trial_args.Kt = trial.suggest_categorical("Kt", [2, 3])
        
        # Suggesting n_his
        trial_args.n_his = trial.suggest_int("n_his", 12, 120, step=6)
        
        # Check if the combination is valid. If not, PRUNE.
        min_required_n_his = trial_args.stblock_num * 2 * (trial_args.Kt - 1) + 1
        if trial_args.n_his < min_required_n_his:
            raise optuna.exceptions.TrialPruned("Invalid architecture.")
        
        ########## End of Preflight pruning ##########
        
        # --- General Training Hyperparameters ---
        trial_args.lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        trial_args.weight_decay_rate = trial.suggest_float("weight_decay_rate", 1e-5, 3e-3, log=True)
        trial_args.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        trial_args.step_size = trial.suggest_int("step_size", 2, 20, step=2)
        trial_args.gamma = trial.suggest_float("gamma", 0.5, 0.95, log=True)
        trial_args.droprate = trial.suggest_float("droprate", 0.1, 0.5)
        trial_args.enable_bias = trial.suggest_categorical("enable_bias", [True, False])
        trial_args.act_func = trial.suggest_categorical("act_func", ["glu", 'gtu', "relu", "silu"])
        
        # --- STGCN Architecture Hyperparameters ---
        if self.args.drop_spatial_layer:
            # Placeholders
            trial_args.graph_conv_type = "none"
            trial_args.Ks = 1
        else:
            trial_args.graph_conv_type = trial.suggest_categorical("graph_conv_type", ["gcn", "cheb"])
            trial_args.Ks = trial.suggest_categorical("Ks", [2, 3])
        trial_args.st_main_channels       = trial.suggest_int("st_main_channels",       32,  128,  step=32)
        trial_args.st_bottleneck_channels = trial.suggest_int("st_bottleneck_channels", 8,   32,   step=8)
        trial_args.output_channels        = trial.suggest_int("output_channels",        128, 1024, step=128)
        
        # We don't tune epochs directly. We set a max value and let early stopping find the best.
        # This is the max number of epochs the model is allowed to run for in each CV fold.
        trial_args.epochs = self.args.epochs
        
        return trial_args
    
    ##########################
    # Setup model
    ##########################
    
    def _setup_model(
            self, 
            args: Any
        ) -> nn.Module:
        """Set up the STGCN model and training components."""
        logger.info(f"Setting up model for the task type '{args.task_type}'...")
        device   = self.input_dict['device']
        logger.info(f"Device: {device}")
        n_nodes = self.input_dict['n_nodes']
        logger.info(f"Number of nodes: {n_nodes}")
        n_features = self.input_dict["n_features"]
        logger.info(f"Number of features: {n_features}")
        
        # Create GSO(s)
        if args.drop_spatial_layer:
            gso = None
        else:
            A = self.input_dict["f_adj_mat_tensor"]
            M = self.input_dict["m_adj_mat_tensors"]
            gso = create_gso(
                args                = args,
                device              = device,
                n_nodes             = n_nodes,
                adj_matrix          = A,
                masked_adj_matrices = M if args.gso_mode == "dynamic" else None,
            )
        
        # Initialize model
        model = HomogeneousSTGCN(
            args        = args,
            n_vertex    = n_nodes,
            n_features  = n_features,
            gso         = gso,
            conv_type   = args.graph_conv_type,
            task_type   = args.task_type,
        ).to(device)

        # Compile model if requested
        if self.args.compile_model:
            logger.info("Compiling model...")
            model = torch.compile(
                model       = model, 
                mode        = args.compile_mode,
                fullgraph   = args.compile_fullgraph,
                dynamic     = args.dynamic_compile,
            )
            logger.info("Model compiled.")
                
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
        
        # 1. Reset model weights, re-initialize optimizer and scheduler for the new fold
        logger.info("Resetting model parameters before training.")
        model.reset_all_parameters(seed=self.seed + fold_index)
        optimizer = create_optimizer(args=trial_params, model=model)
        scheduler = create_scheduler(args=trial_params, optimizer=optimizer)
        
        # 2. Get split payload
        loaders, normalizer = self._get_split_payload(
            args                = trial_params, 
            seed                = self.seed + fold_index,
            train_block_ids     = train_block_ids, 
            val_block_ids       = val_block_ids, 
            test_block_ids      = [])
        
        # 3. Training
        trained_model, history, training_result = train_model(
            args            = trial_params, 
            device          = self.input_dict['device'],
            model           = model, 
            optimizer       = optimizer, 
            scheduler       = scheduler, 
            train_loader    = loaders['train_loader'], 
            val_loader      = loaders['val_loader'],
            trial           = trial,
            epoch_offset    = fold_index * trial_params.epochs)
                
        return training_result
    
    ##########################
    # Cleanup
    ##########################
    
    def _cleanup_after_fold(self):
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        after = torch.cuda.memory_reserved()
        logger.info(f"[fold-cleanup] freed {(before-after)/1024**2:.1f} MiB (reserved {after/1024**2:.1f} MiB)")

    def _cleanup_after_trial(self):
        torch.cuda.synchronize()
        before_r = torch.cuda.memory_reserved()
        before_a = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        
        # Clear Dynamo & Inductor caches
        try: torch._dynamo.reset()
        except Exception: pass
        try: torch._inductor.codecache.clear()
        except Exception: pass
        try: torch._inductor.utils.free_runtime_memory()
        except Exception: pass
        after_r = torch.cuda.memory_reserved()
        after_a = torch.cuda.memory_allocated()
        logger.info(f"[trial-cleanup] reserved freed {(before_r-after_r)/2**20:.1f} MiB; "
                    f"allocated freed {(before_a-after_a)/2**20:.1f} MiB")
    
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
            final_params.epochs = int(np.ceil(epochs * self.args.final_epoch_multiplier))
            logger.info(f"Inferred optimal epochs from CV: {epochs:.0f}. "
                        f"Training final model for {final_params.epochs} rounds (1.1x).")
        
        # Final seed
        # NOTE: For final training, using a different seed,
        #       *2 seems high enough to not clash with the seeds of any previous folds.
        final_seed = self.seed * 2  

        # Get split payload
        loaders, normalizer = self._get_split_payload(
            args                = final_params, 
            seed                = final_seed,
            train_block_ids     = train_block_ids, 
            val_block_ids       = val_block_ids, 
            test_block_ids      = test_block_ids)
                
        # Reset model weights, re-initialize optimizer and scheduler for the new fold
        logger.info("Resetting model parameters before training.")
        model.reset_all_parameters(seed=final_seed)
        optimizer = create_optimizer(args=final_params, model=model)
        scheduler = create_scheduler(args=final_params, optimizer=optimizer)
        
        # Training
        trained_model, history, training_result = train_model(
            args            = final_params, 
            device          = self.input_dict['device'],
            model           = model, 
            optimizer       = optimizer, 
            scheduler       = scheduler, 
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
            args        = final_params, 
            device      = self.input_dict['device'],
            model       = trained_model, 
            test_loader = loaders['test_loader'], 
            normalizer  = normalizer, 
            threshold   = threshold
        )
        
        return trained_model, history, metrics, model_outputs
    
def main():
    """
    Main entry point to run a single experiment.
    
    This function parses command-line arguments and starts the STGCNExperimentRunner.
    """
    from ..config.args import parse_args
    args = parse_args()
    
    if args.tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    runner = STGCNExperimentRunner(args)
    runner.run()

if __name__ == '__main__':
    main()