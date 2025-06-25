import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Logging setup
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class LGBMWrapper(BaseEstimator, RegressorMixin, ClassifierMixin):
    """
    A scikit-learn–compatible wrapper around LightGBM that can do either
    regression or binary classification, based on `objective`.
    """

    def __init__(
        self,

        # Core LightGBM parameters
        objective: str = "regression",  # "regression" or "binary"
        metric: str = "mae",            # e.g. "mae", "rmse", "binary_logloss", "auc"
        boosting_type: str = "gbdt",
        n_estimators: int = 1000,
        random_state: int = 2658918,
        verbosity: int = 1,
        n_jobs: int = -1,

        # Early stopping
        early_stopping_rounds: Optional[int] = 50,

        # Tree structure
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        min_split_gain: float = 0.0,

        # Regularization
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,

        # Sampling & feature selection
        feature_fraction: float = 0.9, # or "colsample_bytree"
        bagging_fraction: float = 0.8, # or "subsample"
        bagging_freq: int = 5,

        # Learning control
        learning_rate: float = 0.05,
        boost_from_average: bool = True,
        
        # Optional for Classification
        is_unbalance: Optional[bool] = True,

        # Extra kwargs
        **kwargs: Any,
    ):
        # Core LightGBM parameters
        self.objective = objective
        self.metric = metric
        self.boosting_type = boosting_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs

        # Early stopping
        self.early_stopping_rounds = early_stopping_rounds

        # Tree structure
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain

        # Regularization
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

        # Sampling & feature selection
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq

        # Learning control
        self.learning_rate = learning_rate
        self.boost_from_average = boost_from_average

        # Optional for Classification
        self.is_unbalance = is_unbalance

        # Extra kwargs
        self.kwargs = kwargs

        # Attributes set at fit time
        self.model_: Union[LGBMRegressor, LGBMClassifier, None] = None
        self.feature_names_: Optional[List[str]] = None
        self.best_iteration_: Optional[int] = None
        self.evals_result_: Dict[str, Any] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[
            List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]
        ] = None,
        eval_names: Optional[List[str]] = None,
        callbacks: Optional[List[Any]] = None,
        verbose: bool = True,
    ) -> "LGBMWrapper":
        """
        Fit either a regressor or classifier depending on `self.objective`.

        Args:
            X: training features
            y: training target
            eval_set: list of (X_val, y_val) for early stopping
            callbacks: list of LightGBM callbacks
            verbose: whether to print progress
        """
        if verbose and callbacks:
            logger.warning(
                "[LGBMWrapper] `verbose=True` has no effect because `callbacks` were manually provided. "
                "To enable logging, include `log_evaluation()` in the callbacks."
            )

        # Validate shapes
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Capture feature names
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None

        # Common params dict
        params = {
            # Core task settings
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            "n_jobs": self.n_jobs,

            # Tree structure
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "min_split_gain": self.min_split_gain,

            # Regularization
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,

            # Sampling & feature selection
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,

            # Learning control
            "learning_rate": self.learning_rate,
            "boost_from_average": self.boost_from_average,

            **self.kwargs,
        }
        
        # Choose appropriate LGBM class
        if self.objective.startswith("binary") or self.objective in ("binary",):
            ModelClass = LGBMClassifier
            params["is_unbalance"] = self.is_unbalance
        else:
            ModelClass = LGBMRegressor
        self.model_ = ModelClass(**params)

        # Build fit_kwargs
        fit_kwargs: Dict[str, Any] = {}
        if eval_names is not None:
            fit_kwargs["eval_names"] = eval_names

        if callbacks is not None:
            fit_kwargs["callbacks"] = callbacks
        elif verbose and eval_set:
            # default logging + early stopping
            fit_kwargs["callbacks"] = [
                log_evaluation(period=5),
                early_stopping(stopping_rounds=self.early_stopping_rounds or 50,
                               first_metric_only=True),
            ]
        
        # Fit underlying model
        self.model_.fit(X, y, eval_set=eval_set, **fit_kwargs)

        # Capture best iteration and evals result
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        self.evals_result_ = getattr(self.model_, "evals_result_", {})

        logger.info(f"[LGBMWrapper] Training complete. Best iter: {self.best_iteration_}")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict using the trained model.
            - For regression: returns numeric predictions.
            - For classification: returns class labels.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")
        return self.model_.predict(X, num_iteration=self.best_iteration_)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Classification-only: returns probability of the positive class.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")
        if not hasattr(self.model_, "predict_proba"):
            raise ValueError("Underlying model does not support predict_proba")
        return self.model_.predict_proba(X, num_iteration=self.best_iteration_)[:, 1]

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")
        imp = self.model_.booster_.feature_importance(importance_type=importance_type)
        features = (
            self.feature_names_
            if self.feature_names_ is not None
            else [f"f{i}" for i in range(len(imp))]
        )
        return (
            pd.DataFrame({"feature": features, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"[LGBMWrapper] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LGBMWrapper":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError(f"File at {path} is not a LGBMWrapper instance")
        logger.info(f"[LGBMWrapper] Model loaded from {path}")
        return model

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        params.pop("kwargs", None)
        return params

    def set_params(self, **params: Any) -> "LGBMWrapper":
        return super().set_params(**params)