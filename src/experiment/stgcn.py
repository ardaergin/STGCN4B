#!/usr/bin/env python

from copy import deepcopy
from typing import Any, Dict, List, Tuple
from argparse import Namespace
from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import optuna
import gc
import torch._dynamo
import torch._inductor.codecache
import torch._inductor.utils
from torch_geometric.data import HeteroData

# Base runner class
from .base import BaseExperimentRunner
# Data Preparation
from ..preparation.preparer import Homogeneous as HomogeneousSTGCNDataPreparer
from ..preparation.preparer import Heterogeneous as HeterogeneousSTGCNDataPreparer
# Normalizers
from ..models.STGCN4B.normalizer import STGCNNormalizer
from ..models.STGCN4B.normalizer import Homogeneous as HomogeneousSTGCNNormalizer
from ..models.STGCN4B.normalizer import Heterogeneous as HeterogeneousSTGCNNormalizer
# Data Loaders
from ..models.STGCN4B.loader import get_data_loaders
# Models
from ..models.STGCN4B.homogeneous.model import HomogeneousSTGCN
from ..models.STGCN4B.heterogeneous.model import HeterogeneousSTGCN
# Utils
from ..models.STGCN4B.gso_utils import create_gso
from ..models.STGCN4B.optim_utils import create_optimizer, create_scheduler
# Training
from ..models.STGCN4B.train import train_model, evaluate_model
from ..utils.tracking import TrainingResult, TrainingHistory

import logging; logger = logging.getLogger(__name__)


class STGCNExperimentRunner(BaseExperimentRunner, ABC):
    def __init__(self, args: Any):        
        super().__init__(args)

    @abstractmethod
    def _get_data_preparer_class(self) -> Any:
        pass

    @abstractmethod
    def _get_normalizer_class(self) -> Any:
        pass

    def _prepare_data(self) -> Dict[str, Any]:
        logger.info("Handling data preparation for STGCN...")
        data_preparer_cls = self._get_data_preparer_class()
        data_preparer = data_preparer_cls(self.args)
        input_dict = data_preparer.get_input_dict()
        return input_dict
    
    #########################
    # Split preparation
    #########################

    @property
    @abstractmethod
    def all_X(self) -> Any:
        """An abstract property that returns the complete feature dataset."""
        pass

    @abstractmethod
    def _slice_train_features(self, train_indices: List[int]) -> Any:
        """Slices the feature data for training."""
        pass
    
    def _normalize_split(
            self, 
            args: Namespace,
            train_block_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, STGCNNormalizer]:
        """
        Normalizes the feature and target arrays for a given split.
        
        VERY IMPORTANT: Must deep copy at this stage, so that the rest of the pipeline
        does not modify the original data. Since all the other operations (imputation,
        data loading, etc.) are all in-place operations.
        """
        # 1. Get train indices and slice arrays to fit the normalizer
        train_indices = self.splitter._get_indices_from_blocks(train_block_ids)
        train_feature_slice = self._slice_train_features(train_indices)
        
        # 3. Fit normalizer and transform
        normalizer_cls = self._get_normalizer_class()
        normalizer = normalizer_cls(args)
        
        # Features
        normalizer.fit_features(
            train_data              = train_feature_slice,
            feature_names           = self.input_dict["feature_names"],
            method                  = args.normalization_method,
            features_to_skip_norm   = args.skip_normalization_for
        )
        norm_features = normalizer.transform_features(all_data=self.all_X)
        
        # Subclass-specific logging
        self._log_normalization_stats(x=norm_features)
        
        # Targets
        train_target_slice = self.input_dict["target_array"][train_indices]
        train_mask_slice = self.input_dict["target_mask"][train_indices]
        normalizer.fit_target(
            train_targets           = train_target_slice, 
            train_mask              = train_mask_slice,
            method                  = 'median'
        )
        norm_target = normalizer.transform_target(targets=self.input_dict["target_array"])
        
        return norm_features, norm_target, normalizer
    
    @abstractmethod
    def _log_normalization_stats(self, x):
        """Abstract method for logging normalization statistics."""
        pass
    
    @abstractmethod
    def _load_data_to_tensors(self, features: Any, targets: np.ndarray) -> Dict[str, Any]:
        """
        Converts features and targets to a dictionary of tensors.

        Subclasses must return a dictionary with keys: 
        - "features":         Tensor of features
        - "targets":          Tensor of targets
        - "target_mask":      Tensor of target masks
        - "target_source":    Tensor of target sources
        """
        pass
    
    def _get_split_payload(
            self, 
            args: Any,
            seed: int,
            train_block_ids: List[int], 
            val_block_ids: List[int], 
            test_block_ids: List[int]
            )-> Tuple[Dict, STGCNNormalizer]:
        """
        Normalize -> Impute -> Load to tensors -> Get data loaders.
        
        **Note**:
        Normalization must not impact the underlying data. 
        But, imputation should be in-place to save memory, 
        since we would have already copied while normalizing.
        """
        # Normalization & Imputation
        norm_features, norm_target_array, normalizer = self._normalize_split(args, train_block_ids)
        features, target_array = self._impute_split(norm_features, norm_target_array)
        
        # Load the processed arrays into tensors
        tensors = self._load_data_to_tensors(features=features, targets=target_array)
        
        # Get DataLoaders
        loaders = get_data_loaders(
            args                        = args,
            seed                        = seed,
            blocks                      = self.input_dict["blocks"],
            target_tensor               = tensors["targets"],
            target_mask_tensor          = tensors["target_mask"],
            target_source_tensor        = tensors["target_source"],
            max_horizon                 = max(args.forecast_horizons),
            train_block_ids             = train_block_ids,
            val_block_ids               = val_block_ids,
            test_block_ids              = test_block_ids,
            graph_type                  = self.args.graph_type,
            feature_data                = tensors["features"],
        )
        return loaders, normalizer
        
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
    
    def _compile_model(self, args: Namespace, model: nn.Module):
        logger.info(">>> Compiling model. <<<")
        compiled_model = torch.compile(
            model       = model, 
            mode        = args.compile_mode,
            fullgraph   = args.compile_fullgraph,
            dynamic     = args.dynamic_compile_mode,
        )
        return compiled_model
    
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
        threshold:          float = None, # legacy
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
                
        # Evaluation
        metrics, model_outputs = evaluate_model(
            args        = final_params, 
            device      = self.input_dict['device'],
            model       = trained_model, 
            test_loader = loaders['test_loader'], 
            normalizer  = normalizer, 
        )
        
        return trained_model, history, metrics, model_outputs



class Homogeneous(STGCNExperimentRunner):
    def __init__(self, args: Any):        
        super().__init__(args)
        
    def _get_data_preparer_class(self):
        return HomogeneousSTGCNDataPreparer

    def _get_normalizer_class(self):
        return HomogeneousSTGCNNormalizer

    #########################
    # Split preparation
    #########################
    
    @property
    def all_X(self) -> np.ndarray:
        """Returns the homogeneous feature array."""
        return self.input_dict["feature_array"]
    
    def _slice_train_features(self, train_indices: List[int]) -> np.ndarray:
        """Slices the homogeneous feature array using the 'all_X' property."""
        return self.all_X[train_indices]
    
    def _log_normalization_stats(self, x: np.ndarray):
        """Logs per-feature statistics for the normalized homogeneous feature array."""
        logger.info("Normalized homogeneous features stats (per-feature):")
        
        feature_names_list = self.input_dict["feature_names"]
        num_features = x.shape[2]

        for i in range(num_features):
            name = feature_names_list[i] if i < len(feature_names_list) else f"Feature_{i}"
            feature_data = x[:, :, i]

            min_val = np.nanmin(feature_data)
            max_val = np.nanmax(feature_data)
            mean_val = np.nanmean(feature_data)
            std_val = np.nanstd(feature_data)

            logger.info(
                f"  Feature '{name:<25}': "
                f"min={min_val:<10.4f}, max={max_val:<10.4f}, "
                f"mean={mean_val:<10.4f}, std={std_val:<10.4f}"
            )
    
    def _impute_split(
            self, 
            feature_array: np.ndarray, 
            target_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-imputes NaN values in the feature and target arrays. In-place operation.
        
        Note: For forecast tasks, we impute to target due to NaN values formed during the target creation.
        This is fine, since we will mask these imputations later.
        """
        clean_feature_array = np.nan_to_num(feature_array, nan=0.0).astype(np.float32, copy=False)
        clean_target_array = np.nan_to_num(target_array, nan=0.0).astype(np.float32, copy=False)
        return clean_feature_array, clean_target_array

    def _load_data_to_tensors(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, torch.Tensor]:
        """Converts a dictionary of numpy arrays to a dictionary of tensors."""
        numpy_payload = {
            "features":          features,
            "targets":           targets,
            "target_mask":      self.input_dict["target_mask"],
            "target_source":    self.input_dict["target_source_array"]
        }
        return {
            name: torch.from_numpy(arr).float()
            for name, arr in numpy_payload.items()
        }
    
    #########################
    # HPO
    #########################
    
    def _suggest_hyperparams(
            self, 
            trial: optuna.trial.Trial
    ) -> Namespace:
        """
        The objective function for Optuna, performing k-fold cross-validation.

        **Note**: 
        - We don't tune epochs directly. We set a max value and let early stopping find the best.
        This is the max number of epochs the model is allowed to run for in each CV fold.
        - This has been removed, since n_his is high enough, but a preflight pruning might be necessary,
        in order to have a valid architecture:
        ```
        min_required_n_his = trial_args.stblock_num * 2 * (trial_args.Kt - 1) + 1
        if trial_args.n_his < min_required_n_his:
            raise optuna.exceptions.TrialPruned("Invalid architecture.")
        ```
        """
        trial_args = deepcopy(self.args)
        
        # Max epochs, early stopping, and pruning setup
        trial_args.epochs               = 20
        trial_args.es_patience          = 5
        trial_args.n_startup_trials     = 5
        trial_args.n_warmup_steps       = 5
        trial_args.interval_steps       = 1
                
        # General Training Hyperparameters
        trial_args.lr                   = trial.suggest_float("lr", 0.0001, 0.01, log=True)
        trial_args.weight_decay_rate    = trial.suggest_float("weight_decay_rate", 0.0001, 0.01, log=True)
        trial_args.droprate             = trial.suggest_float("droprate", 0.05, 0.3)
        trial_args.optimizer            = "adamw"
        trial_args.step_size            = 50
        trial_args.gamma                = 0.99
        trial_args.enable_bias          = True
        trial_args.act_func             = "glu"
        
        # Model architecture
        trial_args.gso_type             = trial.suggest_categorical("gso_type", ["rw_renorm_adj", "col_renorm_adj"])
        trial_args.stblock_num          = trial.suggest_int("stblock_num", 2, 4)
        trial_args.Kt                   = 3
        trial_args.n_his                = 48
        
        base_channels                   = trial.suggest_int("base_channels", 32, 128, step=32)
        trial_args.st_main_channels     = base_channels
        
        bottleneck_factor                 = trial.suggest_float("bottleneck_factor", 0.25, 0.5, step=0.25)
        trial_args.st_bottleneck_channels = max(8, int(base_channels * bottleneck_factor))
        
        output_factor                   = trial.suggest_float("output_factor", 1.0, 3.0, step=1.0)
        trial_args.output_channels      = min(256, int(base_channels * output_factor))
        
        # For turning homogeneous STGCN into TCN for ablation:
        if self.args.drop_spatial_layer:
            # Placeholders:
            trial_args.graph_conv_type = "none"
            trial_args.Ks = 0
        else:
            trial_args.graph_conv_type = "gcn"
            trial_args.Ks = 2
        
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
        n_features += 1 # Due to padding
        logger.info(f"Number of features: {n_features}")
        
        # Create GSO(s)
        if args.drop_spatial_layer:
            gso = None
        else:
            A = self.input_dict["f_adj_mat_tensor"]
            M = self.input_dict["f_masked_adj_mat_tensors"]
            gso = create_gso(
                args                = args,
                device              = device,
                n_nodes             = n_nodes,
                adj_matrix          = A,
                masked_adj_matrices = M if args.gso_mode == "dynamic" else None,
                return_format       = "dense",
                transpose           = args.transpose_gso
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
        if self.args.compile_model: 
            model = self._compile_model(args, model)
        
        return model



class Heterogeneous(STGCNExperimentRunner):
    def __init__(self, args: Any):        
        super().__init__(args)
    
    def _get_data_preparer_class(self):
        return HeterogeneousSTGCNDataPreparer
    
    def _get_normalizer_class(self):
        return HeterogeneousSTGCNNormalizer
    
    #########################
    # Split preparation
    #########################

    @property
    def all_X(self) -> Dict[int, HeteroData]:
        """
        Returns the complete feature dataset, which is a dictionary mapping
        a time index to a HeteroData graph snapshot.
        """
        return self.input_dict["temporal_graphs"]

    def _slice_train_features(self, train_indices: List[int]) -> Dict[int, HeteroData]:
        """
        Slices the dictionary of graphs by creating a new dictionary containing
        only the keys (time indices) present in the training split.
        """
        return {idx: self.all_X[idx] for idx in train_indices}
    
    def _log_normalization_stats(self, x: Dict[int, HeteroData]):
        """Logs per-feature statistics for the normalized heterogeneous graph snapshots."""
        logger.info("Normalized heterogeneous features stats (per-feature):")
        tensors_by_nodetype = defaultdict(list)
        for snapshot in x.values():
            for node_type in snapshot.node_types:
                if 'x' in snapshot[node_type] and snapshot[node_type].x is not None:
                    tensors_by_nodetype[node_type].append(snapshot[node_type].x)

        for node_type, tensor_list in tensors_by_nodetype.items():
            if not tensor_list:
                logger.info(f"  - Node Type '{node_type}': No feature tensors found.")
                continue

            combined_tensor = torch.cat(tensor_list, dim=0)
            feature_names_list = self.input_dict["feature_names"].get(node_type, [])
            num_features = combined_tensor.shape[1]

            logger.info(f"  --- Stats for Node Type: '{node_type}' ({num_features} features) ---")

            for i in range(num_features):
                name = feature_names_list[i] if i < len(feature_names_list) else f"Feature_{i}"
                feature_col = combined_tensor[:, i]
                
                valid_values = feature_col[~torch.isnan(feature_col)]

                if valid_values.numel() > 0:
                    min_val, max_val, mean_val, std_val = (
                        torch.min(valid_values).item(),
                        torch.max(valid_values).item(),
                        torch.mean(valid_values).item(),
                        torch.std(valid_values).item()
                    )
                    logger.info(
                        f"    Feature '{name:<20}': "
                        f"min={min_val:<10.4f}, max={max_val:<10.4f}, "
                        f"mean={mean_val:<10.4f}, std={std_val:<10.4f}"
                    )
                else:
                    logger.info(f"    Feature '{name:<20}': All NaN values.")
    
    def _impute_split(
            self,
            feature_graphs: Dict[int, HeteroData],
            target_array: np.ndarray
    ) -> Tuple[Dict[int, HeteroData], np.ndarray]:
        """
        Imputes NaNs in heterogeneous features (in-place) and the target array.
        """
        # Impute PyTorch feature tensors in-place
        for snapshot in feature_graphs.values():
            for node_type in snapshot.node_types:
                if 'x' in snapshot[node_type]:
                    torch.nan_to_num_(snapshot[node_type].x, nan=0.0)

        # Impute NumPy target array
        clean_target_array = np.nan_to_num(target_array, nan=0.0)
        
        return feature_graphs, clean_target_array

    def _load_data_to_tensors(self, features: Dict[int, HeteroData], targets: np.ndarray) -> Dict[str, Any]:
        """Converts a dictionary of numpy arrays to a dictionary of tensors."""
        numpy_payload = {
            "targets":           targets,
            "target_mask":      self.input_dict["target_mask"],
            "target_source":    self.input_dict["target_source_array"]
        }
        tensors_dict = {
            name: torch.from_numpy(arr).float()
            for name, arr in numpy_payload.items()
        }
        # Merge HeteroData graphs back, they are already tensorized
        tensors_dict["features"] = features
        
        return tensors_dict

    #########################
    # HPO
    #########################
    
    def _suggest_hyperparams(
            self, 
            trial: optuna.trial.Trial
    ) -> Namespace:
        """
        The objective function for Optuna, performing k-fold cross-validation.

        **Note**: 
        - We don't tune epochs directly. We set a max value and let early stopping find the best.
        This is the max number of epochs the model is allowed to run for in each CV fold.
        - This has been removed, since n_his is high enough, but a preflight pruning might be necessary,
        in order to have a valid architecture:
        ```
        min_required_n_his = trial_args.stblock_num * 2 * (trial_args.Kt - 1) + 1
        if trial_args.n_his < min_required_n_his:
            raise optuna.exceptions.TrialPruned("Invalid architecture.")
        ```
        """
        trial_args = deepcopy(self.args)
        
        # Max epochs, early stopping, and pruning setup
        trial_args.epochs               = 20
        trial_args.es_patience          = 5
        trial_args.n_startup_trials     = 5
        trial_args.n_warmup_steps       = 5
        trial_args.interval_steps       = 1
        
        # General Training Hyperparameters
        trial_args.lr                   = trial.suggest_float("lr", 0.0001, 0.01, log=True)
        trial_args.weight_decay_rate    = trial.suggest_float("weight_decay_rate", 0.0001, 0.01, log=True)
        trial_args.droprate             = trial.suggest_float("droprate", 0.05, 0.3)
        trial_args.optimizer            = "adamw"
        trial_args.step_size            = 50
        trial_args.gamma                = 0.99
        trial_args.enable_bias          = True
        trial_args.act_func             = "glu"
        
        # Model Architecture
        trial_args.stblock_num          = 2
        trial_args.Kt                   = 3
        trial_args.n_his                = 48
        
        ## Graph Convolution
        trial_args.gconv_type_p2d       = 'sage'
        trial_args.gconv_type_d2r       = 'sage'
        trial_args.bidir_d2r            = trial.suggest_categorical("bidir_d2r", [True, False])
        trial_args.bidir_p2d            = trial.suggest_categorical("bidir_p2d", [True, False])
        trial_args.aggr_type            = trial.suggest_categorical("aggr_type", ["sum", "mean"])

        trial_args.gso_type             = trial.suggest_categorical("gso_type", ["rw_renorm_adj", "col_renorm_adj"])
        
        ## Per-type channels:
        # - Room:               high
        # - Device/Property:    moderate
        # - Time/Outside:       small
        
        def _clamp(x, low=8, high=128):
            return int(max(low, min(high, round(x))))
        
        def shrink(x): 
            return _clamp(x * out_shrink)

        mid_base                        = trial.suggest_categorical("mid_base", [16, 32, 64])
        trial_args.device_embed_dim     = mid_base
        room_factor                     = trial.suggest_float("room_factor", 1.0, 1.5, step=0.5) # expand
        globals_factor                  = trial.suggest_float("globals_factor", 0.25, 0.5, step=0.25) # shrink
        out_shrink                      = 1 # trial.suggest_float("out_shrink", 0.25, 0.75, step=0.25)
        
        ### Mid
        trial_args.ch_room_mid          = _clamp(room_factor     * mid_base)
        trial_args.ch_device_mid        = _clamp(1.0             * mid_base)
        trial_args.ch_property_mid      = _clamp(1.0             * mid_base)
        trial_args.ch_time_mid          = _clamp(globals_factor  * mid_base)
        trial_args.ch_outside_mid       = _clamp(globals_factor  * mid_base)
        
        ### Output per-type channels
        trial_args.ch_room_out          = shrink(trial_args.ch_room_mid)
        trial_args.ch_device_out        = shrink(trial_args.ch_device_mid)
        trial_args.ch_property_out      = shrink(trial_args.ch_property_mid)
        trial_args.ch_time_out          = shrink(trial_args.ch_time_mid)
        trial_args.ch_outside_out       = shrink(trial_args.ch_outside_mid)
        
        ## Main output
        trial_args.output_channels      = trial_args.ch_room_out # trial.suggest_int("output_channels", 32, 128, step=32)
        
        return trial_args
    
    ##########################
    # Setup model
    ##########################
    
    def _get_gsos(self, args: Any):
        """Small helper for neatness."""
        device = self.input_dict["device"]
        n_room_nodes = self.input_dict['n_nodes']
        logger.info(f"Number of room nodes: {n_room_nodes}")
        
        # Horizontal
        A_h = self.input_dict["h_adj_mat_tensor"]
        M_h = self.input_dict["h_masked_adj_mat_tensors"]
        
        gso_h = create_gso(
            args                = args,
            device              = device,
            n_nodes             = n_room_nodes,
            adj_matrix          = A_h,
            masked_adj_matrices = M_h if args.gso_mode == "dynamic" else None,
            return_format       = "coo"
        )
        if args.gso_mode == "static":
            gso_h = [gso_h] * args.stblock_num

        # Vertical
        A_v = self.input_dict["v_adj_mat_tensor"]
        M_v = self.input_dict["v_masked_adj_mat_tensors"]
        
        gso_v = create_gso(
            args                = args,
            device              = device,
            n_nodes             = n_room_nodes,
            adj_matrix          = A_v,
            masked_adj_matrices = M_v if args.gso_mode == "dynamic" else None,
            return_format       = "coo"
        )
        if args.gso_mode == "static":
            gso_v = [gso_v] * args.stblock_num

        return gso_h, gso_v
    
    def _get_outside_ei_and_ew(self):
        outside_adj_vector = self.input_dict["o_adj_vec_tensor"]
        outside_idx = 0
        
        # Find the indices of all rooms that have a connection from the outside
        room_idx = np.nonzero(outside_adj_vector > 0)[0]
        
        # Get the corresponding weights for those connections
        edge_weights = outside_adj_vector[room_idx]
        edge_weights = edge_weights.unsqueeze(1)
        
        # Convert to PyTorch tensors and assign to the hetero_data object
        edge_index = torch.vstack((
            torch.full_like(torch.as_tensor(room_idx), outside_idx, dtype=torch.long),
            torch.as_tensor(room_idx, dtype=torch.long),
        ))  # shape (2, E)
        logger.info(f"Added {len(room_idx)} weighted outside-to-room edges")
        return edge_index, edge_weights
        
    def _setup_model(
            self, 
            args: Any
        ) -> nn.Module:
        """Set up the Heterogeneous STGCN model and training components."""
        device = self.input_dict["device"]
        
        # 1.  Metadata & feature dimensions
        base_graph: HeteroData = self.input_dict["base_graph"] # single, static snapshot
        metadata = base_graph.metadata() # (node_types, edge_types)
        
        # 2. Get per‑node‑type input‑channel counts
        feat_names = self.input_dict["feature_names"]
        node_feature_dims = {
            ntype: len(feat_names[ntype])
            for ntype in feat_names
        }
        logger.info(f"Model input feature dimensions: {node_feature_dims}")
        
        # 3. Prepare Edge Information for ALL ST-Blocks
        
        # 3A. Edges that stay the same throughout ST-blocks
        
        # Get outside edge
        outside_edge_index, outside_edge_weight = self._get_outside_ei_and_ew()
        
        # Build the static base dictionary
        static_edges_base = {}
        for edge_type in base_graph.edge_types:
            # Room to Room edges
            if edge_type == ('room', 'adjacent_horizontal', 'room'):
                continue
            if edge_type == ('room', 'adjacent_vertical', 'room'):
                continue
            # Outside to Room edges
            elif edge_type == ('outside', 'influences', 'room'):
                static_edges_base[edge_type] = {
                    'index': outside_edge_index.to(device),
                    'weight': outside_edge_weight.to(device)
                }
            # All other edges (all of them are non-weighted)
            else:
                edge_store = base_graph[edge_type]
                static_edges_base[edge_type] = {
                    'index': edge_store.edge_index.to(device),
                    'weight': None
                }
        
        # 3B. Room to room edges that change throughout ST-blocks
        gso_h, gso_v = self._get_gsos(args)
        
        # 3C. Combine base edges with per-block room edges
        all_edges_by_block = []
        for i in range(args.stblock_num):
            # Non-room edges
            edges_for_this_block = deepcopy(static_edges_base)
            # Room edges: Horizontal
            ei_h, ew_h = gso_h[i]
            edges_for_this_block[('room', 'adjacent_horizontal', 'room')] = {'index': ei_h, 'weight': ew_h}
            # Room edges: Vertical
            ei_v, ew_v = gso_v[i]
            edges_for_this_block[('room', 'adjacent_vertical', 'room')] = {'index': ei_v, 'weight': ew_v}
            
            all_edges_by_block.append(edges_for_this_block)
        
        # 4. Instantiate the model
        model = HeterogeneousSTGCN(
            args                = args,
            metadata            = metadata,
            all_edges_by_block  = all_edges_by_block,
            node_feature_dims   = node_feature_dims,
            property_types      = self.input_dict["property_types"],
            num_devices         = self.input_dict["num_devices"],
            task_type           = args.task_type,
        ).to(device)
        if self.args.compile_model: 
            model = self._compile_model(args, model)
        
        return model


def main():
    """
    Main entry point to run a single experiment.
    
    This function parses command-line arguments and starts the STGCNExperimentRunner.
    """
    from ..config.args import parse_args
    args = parse_args()
    
    # tf32 settings:
    if args.tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Homogeneous or Heterogeneous
    if args.graph_type == "homogeneous":
        runner = Homogeneous(args)
        logger.info("Running Homogeneous STGCN experiment.")
    elif args.graph_type == "heterogeneous":
        runner = Heterogeneous(args)
        logger.info("Running Heterogeneous STGCN experiment.")
    else:
        raise ValueError(f"Unknown graph type: {args.graph_type}.")
    
    # Run the experiment
    runner.run()

if __name__ == '__main__':
    main()