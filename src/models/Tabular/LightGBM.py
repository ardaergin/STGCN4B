import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# Classification Metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix, 
    ConfusionMatrixDisplay)
# Regression Metrics
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score)
import optuna

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from ...data.Loader.tabular_dataset import TabularDataset

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
        objective: str = "regression",  # "regression" or "binary"
        metric: str = "mae",            # e.g. "mae", "rmse", "binary_logloss", "auc"
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        random_state: int = 42,
        verbosity: int = 1,
        n_estimators: int = 1000,
        early_stopping_rounds: Optional[int] = 50,
        **kwargs: Any,
    ):
        # Core LightGBM parameters
        self.objective = objective
        self.metric = metric
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.random_state = random_state
        self.verbosity = verbosity

        # Train-time parameters
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds

        # Any extra LightGBM params
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
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            "n_estimators": self.n_estimators,
            **self.kwargs,
        }
        
        # Add early_stopping_rounds to params if provided
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        # Choose appropriate LGBM class
        if self.objective.startswith("binary") or self.objective in ("binary",):
            ModelClass = LGBMClassifier
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
        imp = self.model_.feature_importances_(importance_type)
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





class LightGBMTrainer:
    """
    Trainer for LGBMWrapper: handles training, evaluation, and model persistence.
    """

    def __init__(
        self,
        model: LGBMWrapper,
        output_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger = logger or logging.getLogger(__name__)

    def train(
        self,
        dataset: TabularDataset,
        callbacks: Optional[List[Any]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train on the given TabularDataset.  `dataset.task` must be
        either "forecasting" or "classification".
        """
        # ensure the dataset is in a known mode
        if not hasattr(dataset, "task"):
            raise ValueError("Call dataset.set_mode(...) before training")
        task = dataset.task  # "forecasting" or "classification"

        # infer LightGBM objective & metric
        if task == "classification":
            self.model.set_params(objective="binary", metric="binary_logloss")
            X_train, y_train = dataset.X_train, dataset.y_train
            X_val, y_val = dataset.X_val, dataset.y_val
        else:  # forecasting/regression
            self.model.set_params(objective="regression", metric="mae")
            X_train, y_train = dataset.X_train, dataset.y_train
            X_val, y_val = dataset.X_val, dataset.y_val

        # fit
        self.model.fit(
            X_train,
            y_train,
            eval_set=[ (X_train, y_train), (X_val, y_val) ],
            eval_names=["train","valid"],
            callbacks=callbacks,
            verbose=verbose,
        )

        results = {
            "best_iteration": self.model.best_iteration_,
            "evals_result": self.model.evals_result_,
        }
        logger.info(f"[Trainer] Trained ({task}); best iter = {results['best_iteration']}")
        return results

    def evaluate(self, dataset: TabularDataset) -> Dict[str, float]:
        """
        Evaluate on the dataset's test split, choosing metrics by task.
        """
        if not hasattr(dataset, "task"):
            raise ValueError("Call dataset.set_mode(...) before evaluation")
        task = dataset.task
        X_test, y_test = dataset.X_test, dataset.y_test
        preds = self.model.predict(X_test)

        if task == "classification":
            # basic classification metrics
            acc   = accuracy_score(y_test, preds)
            prec  = precision_score(y_test, preds, zero_division=0)
            rec   = recall_score(y_test, preds, zero_division=0)
            f1    = f1_score(y_test, preds, zero_division=0)

            # AUC (if probabilities are available)
            try:
                probs = self.model.predict_proba(X_test)
                auc   = roc_auc_score(y_test, probs)
            except Exception:
                auc = float("nan")

            logger.info(
                f"[Trainer] Eval (cls) -- "
                f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, auc={auc:.4f}"
            )
            return {
                "accuracy":   acc,
                "precision":  prec,
                "recall":     rec,
                "f1":         f1,
                "auc":        auc,
            }

        else:
            # regression metrics
            rmse = root_mean_squared_error(y_test, preds)
            mae  = mean_absolute_error(y_test, preds)
            r2   = r2_score(y_test, preds)
            logger.info(f"[Trainer] Eval (reg) -- rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}")
            return {
                "rmse": rmse,
                "mae":  mae,
                "r2":   r2,
            }

    def save_model(self, filename: str = "model.joblib") -> str:
        path = os.path.join(self.output_dir, filename)
        self.model.save(path)
        logger.info(f"[Trainer] Model saved to {path}")
        return path
    
    @classmethod
    def load_model(cls, path: str) -> LGBMWrapper:
        return LGBMWrapper.load(path)





class LightGBMTuner:
    """
    Optuna-based hyperparameter tuner for LGBMWrapper using a TabularDataset.
    Automatically infers regression vs. binary classification from dataset.task
    and tunes the appropriate objective/metric.
    """

    def __init__(
        self,
        dataset: TabularDataset,
        random_state: int = 42,
    ):
        if not hasattr(dataset, "task"):
            raise ValueError("Call dataset.set_mode(...) before tuning")
        self.dataset = dataset
        self.task = dataset.task
        self.dataset.set_mode(self.task)
        self.X = self.dataset.X
        self.y = self.dataset.y
        self.train_idx = self.dataset.train_idx
        self.val_idx = self.dataset.val_idx
        self.random_state = random_state
        self.study: Optional[optuna.Study] = None

    def _objective(self, trial: optuna.Trial) -> float:
        # sample hyperparameters
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "random_state": self.random_state,
            "verbosity": -1,  # Keep quiet during tuning
        }
        # choose objective & metric
        if self.task == "classification":
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
        else:
            params["objective"] = "regression"
            params["metric"] = "mae"

        # split
        X_train = self.X.iloc[self.train_idx]
        y_train = self.y[self.train_idx]
        X_val = self.X.iloc[self.val_idx]
        y_val = self.y[self.val_idx]

        # train
        model = LGBMWrapper(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # evaluate
        preds = model.predict(X_val)
        if self.task == "classification":
            # use logloss as objective to minimize?
            probs = model.predict_proba(X_val)
            return log_loss(y_val, probs)
        else:
            return root_mean_squared_error(y_val, preds)

    def tune(
        self,
        n_trials: int = 50,
        direction: str = "minimize",
        study_name: str = "lgbm_tuning",
        storage: Optional[str] = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            storage=storage,
            load_if_exists=load_if_exists,
        )
        self.study.optimize(self._objective, n_trials=n_trials)
        logger.info(
            f"[Tuner] Done ▶ best value={self.study.best_value:.4f}, params={self.study.best_params}"
        )
        return self.study

    def get_best_params(self) -> Dict[str, Any]:
        if self.study is None:
            raise RuntimeError("No study found — call `.tune()` first.")
        return self.study.best_params

    def save_study(self, path: str) -> str:
        if not self.study:
            raise ValueError("No study to save")
        joblib.dump(self.study, path)
        logger.info(f"[Tuner] Study saved to {path}")
        return path

    def load_study(self, path: str) -> optuna.Study:
        self.study = joblib.load(path)
        logger.info(f"[Tuner] Study loaded from {path}")
        return self.study





def main():
    """
    Example usage and testing of LightGBM components (ignoring the tuning for now).
    Run this script directly to test the functionality.
    """
    from ...config.args import parse_args
    args = parse_args()
    
    # Load the data
    input_path = os.path.join(args.data_dir, "processed", "tabular_dataset.joblib")
    dataset = TabularDataset.load(input_path)
    
    if args.task_type == 'classification':
        dataset.set_mode("classification")
    elif args.task_type == 'forecasting':
        dataset.set_mode("forecasting")
    
    logger.info(f"X_train shape: {dataset.X_train.shape}")
    logger.info(f"y_train shape: {dataset.y_train.shape}")
    logger.info(f"X_val shape: {dataset.X_val.shape}")
    logger.info(f"y_val shape: {dataset.y_val.shape}")
    logger.info(f"X_test shape: {dataset.X_test.shape}")
    logger.info(f"y_test shape: {dataset.y_test.shape}")
    
    # Example model creation
    model = LGBMWrapper(verbosity=1)
    logger.info("Model creation successful")

    # Trainer
    trainer = LightGBMTrainer(model=model, output_dir=args.output_dir)
    logger.info("Trainer creation successful")
    
    # Train with verbose output
    logger.info("Starting training...")
    result = trainer.train(dataset, verbose=True)
    logger.info(f"Training successful! Best iteration: {result['best_iteration']}")
    
    # Evaluate
    logger.info("Starting evaluation...")
    metrics = trainer.evaluate(dataset)
    logger.info(f"Evaluation successful! Metrics: {metrics}")
    

    ############### Save ############### 
    # Model
    model_path = trainer.save_model("test_model.joblib")
    logger.info(f"✓ Model saved to: {model_path}")
    
    # Metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics written to {metrics_path}")

    # Feature importance
    fi = model.get_feature_importance()
    fi_csv = os.path.join(args.output_dir, "feature_importance.csv")
    fi.to_csv(fi_csv, index=False)
    logger.info(f"Feature importances saved to {fi_csv}")

    # Ploting top N
    top_n = 20
    fig, ax = plt.subplots(figsize=(8,6))
    fi.head(top_n).plot.barh(x="feature", y="importance", ax=ax, legend=False)
    ax.invert_yaxis()
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fi_plot = os.path.join(args.output_dir, "feature_importance.png")
    fig.savefig(fi_plot)
    plt.close(fig)
    logger.info(f"Feature‐importance plot saved to {fi_plot}")

    # Confusion matrix (classification only)
    if dataset.task == "classification":
        preds = model.predict(dataset.X_test)
        cm = confusion_matrix(dataset.y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["off-hour","work-hour"])
        fig, ax = plt.subplots(figsize=(5,5))
        disp.plot(ax=ax, cmap="Blues")
        fig.tight_layout()
        cm_plot = os.path.join(args.output_dir, "confusion_matrix.png")
        fig.savefig(cm_plot)
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {cm_plot}")

    # Raw evaluation results
    er = model.evals_result_
    evals_json = os.path.join(args.output_dir, "evals_result.json")
    with open(evals_json, "w") as f:
        json.dump(er, f, indent=2)
    logger.info(f"Saved raw evals_result to {evals_json}")

    # Also, plotting per-iteration curves for every split/metric
    for split_name, metrics_dict in er.items():
        for metric_name, values in metrics_dict.items():
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(values, label=split_name)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{split_name} {metric_name} per iteration")
            ax.legend()
            fig.tight_layout()
            curve_png = os.path.join(
                args.output_dir,
                f"{split_name}_{metric_name}_curve.png"
            )
            fig.savefig(curve_png)
            plt.close(fig)
            logger.info(f"Saved curve plot: {curve_png}")

    logger.info("\n" + "=" * 20)
    logger.info("All tests passed! Module is working correctly.")


if __name__ == "__main__":
    main()