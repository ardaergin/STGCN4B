#!/usr/bin/env python

from copy import deepcopy
from typing import Any, Dict, List, Tuple
from argparse import Namespace
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
from ..models.STGCN4B.homogeneous.models import HomogeneousSTGCN
from ..models.STGCN4B.homogeneous.setup import create_gso # for Homogeneous
from ..models.STGCN4B.heterogeneous.model import HeterogeneousSTGCN
from ..models.STGCN4B.homogeneous.setup import create_optimizer, create_scheduler # for both
# Training
from ..models.STGCN4B.homogeneous.train import train_model, evaluate_model
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
        normalizer = normalizer_cls()
        
        # Features
        normalizer.fit_features(
            train_data              = train_feature_slice,
            feature_names           = self.input_dict["feature_names"],
            method                  = args.normalization_method,
            features_to_skip_norm   = args.skip_normalization_for
        )
        norm_features = normalizer.transform_features(all_data=self.all_X)
        
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
        """The objective function for Optuna, performing k-fold cross-validation."""
        trial_args = deepcopy(self.args)
        
        ########## Preflight pruning ##########
        # Handling the possible invalid architecture
        trial_args.stblock_num = trial.suggest_int("stblock_num", 2, 4)
        trial_args.Kt = trial.suggest_categorical("Kt", [2, 3])
        
        # Suggesting n_his
        trial_args.n_his = trial.suggest_int("n_his", 12, 72, step=6)
        
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
                
        if self.args.drop_spatial_layer:
            # Placeholders:
            trial_args.graph_conv_type = "none"
            trial_args.Ks = 1
        else:
            trial_args.graph_conv_type = trial.suggest_categorical("graph_conv_type", ["gcn", "cheb"])
            trial_args.Ks = trial.suggest_categorical("Ks", [2, 3])

        trial_args.st_main_channels       = trial.suggest_int("st_main_channels",       16,  96,   step=16)
        trial_args.st_bottleneck_channels = trial.suggest_int("st_bottleneck_channels", 8,   32,   step=8)
        trial_args.output_channels        = trial.suggest_int("output_channels",        64,  256,  step=32)

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
                return_format       = "dense"
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
        """The objective function for Optuna, performing k-fold cross-validation."""
        trial_args = deepcopy(self.args)
        
        ########## Preflight pruning ##########
        # Handling the possible invalid architecture
        trial_args.stblock_num = trial.suggest_int("stblock_num", 2, 3)
        trial_args.Kt = 2
        
        # Suggesting n_his
        trial_args.n_his = 24
        
        # Check if the combination is valid. If not, PRUNE.
        min_required_n_his = trial_args.stblock_num * 2 * (trial_args.Kt - 1) + 1
        if trial_args.n_his < min_required_n_his:
            raise optuna.exceptions.TrialPruned("Invalid architecture.")
        
        ########## End of Preflight pruning ##########
        
        # --- General Training Hyperparameters ---
        trial_args.lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        trial_args.weight_decay_rate = trial.suggest_float("weight_decay_rate", 1e-5, 3e-3, log=True)
        trial_args.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        trial_args.step_size = trial.suggest_int("step_size", 2, 6, step=2)
        trial_args.gamma = trial.suggest_float("gamma", 0.5, 0.7, log=True)
        trial_args.droprate = trial.suggest_float("droprate", 0.1, 0.2)
        trial_args.enable_bias = True
        trial_args.act_func = "glu"
        
        trial_args.st_main_channels       = trial.suggest_int("st_main_channels",       16,  48,   step=8)
        trial_args.output_channels        = trial.suggest_int("output_channels",        32,  64,  step=8)

        # We don't tune epochs directly. We set a max value and let early stopping find the best.
        # This is the max number of epochs the model is allowed to run for in each CV fold.
        trial_args.es_patience = 3
        trial_args.epochs = 10
        trial_args.n_startup_trials = 3
        trial_args.n_warmup_steps = 5
        trial_args.interval_steps = 1
                
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
            node_feature_dims   = node_feature_dims,
            task_type           = args.task_type,
            all_edges_by_block  = all_edges_by_block,
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