#!/usr/bin/env python

from typing import Any, Dict, List, Tuple
from argparse import Namespace
import pandas as pd

from .base import BaseExperimentRunner
from ..preparation.preparer import TabularDataPreparer
from ..models.naive.model import NaivePersistenceModel
from ..utils.tracking import TrainingHistory

import logging; logger = logging.getLogger(__name__)


class NaiveExperimentRunner(BaseExperimentRunner):
    """
    A zero-HPO runner that just trains & evaluates the persistence baseline.
    """

    def __init__(self, args: Any):
        assert args.run_mode == "single_run", "NaiveExperimentRunner is only meant to be used in single_run mode. No CV-HPO allowed." 
        assert args.prediction_type == "absolute", "For NaiveExperimentRunner, absolute and delta predictions are essentially the same." 
        super().__init__(args)
    
    def _prepare_data(self) -> Dict[str, Any]:
        logger.info("Handling data preparation via TabularDataPreparer...")
        data_preparer = TabularDataPreparer(self.args)
        input_dict = data_preparer.get_input_dict()
        return input_dict
    
    #########################
    # Split preparation
    #########################
    
    def _normalize_split(self): pass
    def _impute_split(self): pass
    
    def _get_split_payload(self, block_ids: List[int]) -> Tuple[pd.Series, pd.Series]:
        """
        Indicing is the same as in LGBMExperimentRunner._get_split_payload().

        But we do not need most of the other logic here.
        """
        indices = []
        blocks = self.input_dict['blocks']
        for block_id in block_ids:
            indices.extend(blocks[block_id]['bucket_indices'])
        
        # Filter based on the 'bucket_idx'
        split_df = self.input_dict['df'][self.input_dict['df']['bucket_idx'].isin(indices)].copy()
        
        # Get target (y)
        target_colname = self.input_dict['target_colnames'][0]
        y = split_df[target_colname]

        # Get source (X)
        full_target_source_df = self.input_dict["target_source_df"]

        merge_on_cols = ['bucket_idx']
        if self.args.task_type == "measurement_forecast":
            merge_on_cols.append('room_uri_str')
        merge_keys_df = split_df[merge_on_cols].copy()
        
        # Merge
        merged_df = pd.merge(
            merge_keys_df, full_target_source_df, 
            on=merge_on_cols, how='left'
        )
        X = merged_df[self.input_dict["source_colname"]]
        
        return X, y
    
    #########################
    # HPO-related methods
    #########################
    def _suggest_hyperparams(self, trial):
        raise NotImplementedError("Naive baseline does not use Optuna.")
    def _train_one_fold(self, *args, **kwargs):
        raise NotImplementedError("Naive baseline does not use CV folds.")
    def _cleanup_after_trial(self): pass
    def _cleanup_after_fold(self): pass
    
    ##########################
    # Setup model
    ##########################

    def _setup_model(self, args: Any) -> NaivePersistenceModel:
        model = NaivePersistenceModel(args=args)
        return model
    
    ##########################
    # Final Training & Evaluation
    ##########################
    def _train_and_evaluate_final_model(
        self, 
        model:              NaivePersistenceModel,
        final_params:       Namespace,
        epochs:             int = None,
        threshold:          float = None,
        *,
        train_block_ids:    List[int],
        val_block_ids:      List[int],
        test_block_ids:     List[int],
        ) -> Tuple[Any, TrainingHistory, Dict[str, Any], Dict[str, Any]]:
        
        # No training or validation sets for persistence baseline, just test.
        X, y = self._get_split_payload(block_ids = test_block_ids)
        
        # No training
        trained_model = None
        history = TrainingHistory(train_metric=None, train_objective=None)
        
        # Evaluate test
        metrics, model_outputs = model.evaluate_model(X_test=X, y_test=y)
                
        return trained_model, history, metrics, model_outputs

def main():
    from ..config.args import parse_args
    args = parse_args()
    runner = NaiveExperimentRunner(args)
    runner.run()

if __name__ == '__main__':
    main()