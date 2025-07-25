from typing import Any, Dict, List, Iterable, Literal
from dataclasses import dataclass, field, asdict

import logging; logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    metric:             float                   # the objective reported to Optuna
    best_epoch:         int|None = None         # best epoch/iteration for validation score
    optimal_threshold:  float|None = None       # best threshold for classifiers


@dataclass
class TrainingHistory:
    """Container for per-epoch training / validation curves plus run metadata."""
    # Metadata
    train_metric:       str
    train_objective:    Literal["minimize", "maximize"]
    optuna_metric:      str | None = None
    optuna_objective:   Literal["minimize", "maximize"] | None = None
    # Main dictionaries
    train: Dict[str, List[float]] = field(default_factory=dict)
    valid: Dict[str, List[float]] = field(default_factory=dict)
    
    def log_epoch(self, dataset, **metrics: float) -> None:
        """
        Add one datapoint per metric, per epoch (called inside loop).

        >>> hist.log("train", loss=0.42, auc=0.71)
        """
        store = getattr(self, dataset)
        if store is None:
            raise ValueError(f"Unknown phase {dataset!r}")
        for k, v in metrics.items():
            store.setdefault(k, []).append(float(v))

    def get_best_valid_score(self) -> float | None:
        """
        Return the best score from the validation set for
        1. the optuna metric
        2. if optuna metric None, then the train metric
        """
        metric = self.train_metric        if self.optuna_metric     is None else self.optuna_metric
        objective = self.train_objective  if self.optuna_objective  is None else self.optuna_objective

        values = self.valid.get(metric)
        if not values: return None
        elif objective == "maximize": return max(values)
        else: return min(values)
    
    def flatten_curves(self, datasets: Iterable[str] = ("train", "valid")) -> Dict[str, List[float]]:
        """
        Return a *flat* dict like {"train_loss": [...], "valid_auc": [...]}.
        """
        out: Dict[str, List[float]] = {}
        for dataset in datasets:
            store = self.train if dataset == "train" else self.valid
            for m, vals in store.items():
                out[f"{dataset}_{m}"] = vals
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_lgbm(
        cls,
        evals_result: Dict[str, Dict[str, List[float]]],
        *,
        train_metric: str,
        train_objective: str,
        optuna_metric: str | None = None,
        optuna_objective: str | None = None,
    ) -> "TrainingHistory":
        hist = cls(train_metric, train_objective, optuna_metric, optuna_objective)
        for dataset, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                # LightGBM calls MAE 'l1', RMSE 'rmse', etc. → normalise the names
                canonical = {
                    "l1": "mae", "l2": "mse", "binary_logloss": "logloss"
                }.get(metric_name, metric_name)
                for v in values:
                    hist.log_epoch(dataset, **{canonical: v})
        return hist
