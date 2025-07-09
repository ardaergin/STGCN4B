#!/usr/bin/env python

import os
from copy import deepcopy
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

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


class STGCNSingleRunner:
    """
    Runs a single STGCN experiment with default parameters (no Optuna, no cross-validation).
    """

    def __init__(self, args: Any):
        """
        Initializes the STGCNSingleRunner for a single experiment instance.

        Args:
            args: A configuration object containing all necessary parameters.
        """
        self.args = args
        self.seed = args.seed

        # Set up the specific output directory for this run's artifacts
        self.output_dir = args.output_dir
        logger.info(f"Experiment outputs will be saved in: {self.output_dir}")

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
        if self.args.task_type == "measurement_forecast":
            targets_numpy[np.isnan(targets_numpy)] = 0.0

        # Converting the processed targets to a FloatTensor
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
    
    def run_single_experiment(self):
        """
        Executes a single experiment with default parameters.
        """
        logger.info("Starting single run experiment...")
        all_blocks = self.input_dict["blocks"]
        seed = self.seed

        logger.info(f"===== Starting Single Run Experiment | Seed: {seed} =====")

        # 1. Create a train-test split for this experiment
        splitter = StratifiedBlockSplitter(
            output_dir=self.output_dir, 
            blocks=all_blocks, 
            stratum_size=self.args.stratum_size, 
            seed=seed
        )
        splitter.get_train_test_split()

        # 2. Create a single train-validation split from the training data
        train_ids, val_ids = splitter.get_single_split()

        # 3. Get test ids
        test_ids = splitter.test_block_ids

        # 4. Setup default parameters
        default_params = self._get_default_params()
        
        # 5. Train model with default parameters
        logger.info(f"Training with default parameters: {default_params}")
        
        # Setup data loaders
        loaders, processor = self._normalize_data_and_get_dataloaders(
            self.args, train_ids, val_ids, test_ids, splitter
        )
        data_for_setup = {**self.input_dict, **loaders}
        
        # Setup model
        model, criterion, optimizer, scheduler, early_stopping = setup_model(
            self.args, data_for_setup
        )
        
        # Train model
        trained_model, history = train_model(
            self.args, model, criterion, optimizer, scheduler,
            early_stopping=early_stopping,
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader']
        )

        logger.info(f"Loading best model from epoch {early_stopping.best_epoch} for final evaluation.")
        trained_model.load_state_dict(early_stopping.best_model_state)

        # 6. Evaluate on test set
        if self.args.task_type == "workhour_classification":
            # Find optimal threshold on validation set
            threshold = find_optimal_threshold(trained_model, loaders['val_loader'])
            logger.info(f"Optimal classification threshold: {threshold:.4f}")
            test_metrics = evaluate_model(self.args, trained_model, loaders['test_loader'], 
                                        processor, threshold=threshold)
        else:
            test_metrics = evaluate_model(self.args, trained_model, loaders['test_loader'], 
                                        processor, threshold=None)
        
        # 7. Save results
        handler = ResultHandler(output_dir=self.output_dir, task_type=self.args.task_type,
                                history=history, metrics=test_metrics)
        handler.process()
        
        # Save test metrics
        scalar_metrics = {k: v for k, v in test_metrics.items() if np.isscalar(v)}
        scalar_metrics['seed'] = seed
        
        test_df = pd.DataFrame([scalar_metrics])
        test_path = os.path.join(self.output_dir, "results_test.csv")
        test_df.to_csv(test_path, index=False)
        
        # Log results
        logger.info(f"\n===== TEST METRICS =====\n{scalar_metrics}\n===== =====")
        logger.info(f"\n===== PARAMETERS USED =====\n{default_params}\n===== =====")
        logger.info(f"===== Single Run COMPLETED. Results saved to {self.output_dir}.")
    
    def _get_default_params(self):
        """
        Returns the default parameters for the model.
        These are the default values from the argparse configuration.
        """
        return {
            'lr': self.args.lr,
            'weight_decay_rate': self.args.weight_decay_rate,
            'optimizer': self.args.optimizer,
            'droprate': self.args.droprate,
            'enable_bias': self.args.enable_bias,
            'graph_conv_type': self.args.graph_conv_type,
            'act_func': self.args.act_func,
            'stblock_num': self.args.stblock_num,
            'n_his': self.args.n_his,
            'Kt': self.args.Kt,
            'Ks': self.args.Ks,
            'st_main_channels': self.args.st_main_channels,
            'st_bottleneck_channels': self.args.st_bottleneck_channels,
            'output_channels': self.args.output_channels,
            'epochs': self.args.epochs
        }


def main():
    """
    Main entry point to run a single experiment.
    """
    from ....config.args import parse_args
    args = parse_args()
    
    runner = STGCNSingleRunner(args)
    runner.run_single_experiment()


if __name__ == '__main__':
    main()