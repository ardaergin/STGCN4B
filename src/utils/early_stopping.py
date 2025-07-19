import torch

import logging; logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping utility.

    Args:
    - direction : {"minimize", "maximize"}. Whether lower or higher metric is better.
    - patience : int. Number of epochs to wait without improvement.
    - delta : float. Minimal *meaningful* improvement.
    - verbose : bool. If True, print checkpoint messages.
    """
    def __init__(self, direction: str = "minimize", 
                 patience: int = 10, 
                 delta: float = 0.01,
                 verbose: bool = False):
        assert direction in {"minimize", "maximize"}
        self.direction  = direction
        self.patience   = patience
        self.delta      = delta
        self.verbose    = verbose

        self.counter          = 0
        self.best_score       = None
        self.best_metric      = None
        self.best_epoch       = 0
        self.best_model_state = None
        self.early_stop       = False
    
    def __call__(self, metric: float, model: torch.nn.Module, epoch: int):
        """Update with a new validation metric."""
        # Decide whether the new metric is better
        improved = (
            self.best_metric is None
            or (self.direction == "minimize" and metric < self.best_metric - self.delta)
            or (self.direction == "maximize" and metric > self.best_metric + self.delta)
        )
        
        if improved:
            self._save_checkpoint(metric, model, epoch)
        else:
            self.counter += 1
            if self.verbose:
                logger.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _save_checkpoint(self, metric: float, model: torch.nn.Module, epoch: int):
        """Record the current model & metric as the best so far."""
        self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        self.best_metric = metric
        self.best_epoch = epoch
        self.counter = 0
        if self.verbose:
            logger.info("New best %.6f at epoch %d", metric, epoch)