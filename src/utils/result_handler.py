from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from ..utils.tracking import TrainingHistory

import logging; logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    """Create directory *path* (and parents) if it does not yet exist."""
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ResultHandler:
    """Visualise & persist training curves + evaluation artefacts."""
    def __init__(
        self,
        output_dir: str | Path,
        task_type: str,
        history: TrainingHistory,
        metrics: Dict[str, Any],
        *,
        model_outputs: Dict[str, Any] | None = None,
        model: Any | None = None
    ) -> None:
        self.output_dir = Path(output_dir)
        _ensure_dir(self.output_dir)

        self.task_type = task_type
        self.history = history
        self.metrics = metrics
        self.model_outputs = model_outputs or {}
        self.model = model

        # Detect primary training metric (loss) name
        self._train_metric_name = self.history.train_metric
        logger.info("Using '%s' as primary train metric", self._train_metric_name)

    @property
    def has_history(self) -> bool:
        """Check if there is any meaningful training history to plot."""
        # The history is meaningful if either the train or valid logs are not empty.
        return bool(self.history.train or self.history.valid)
    
    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def process(self) -> None:  # noqa: D401 – imperative mood
        """Create plots + save metrics to disk."""
        logger.info("Generating result artefacts → %s", self.output_dir)

        if self.task_type in {"measurement_forecast", "consumption_forecast"}:
            self._plot_forecasting_main()
            self._plot_forecasting_timeseries()
            self._save_forecasting_metrics()
        elif self.task_type == "workhour_classification":
            self._plot_classification()
            self._save_classification_metrics()
        else:
            raise ValueError(f"Unknown task type '{self.task_type}'.")

        # Optional – feature‑importance plot for e.g. LGBM
        if self.model is not None and hasattr(self.model, "get_feature_importance"):
            self._plot_feature_importance()
        
    # ------------------------------------------------------------------
    # Forecasting helpers
    # ------------------------------------------------------------------

    def _get_series(self, dataset: str, metric: str | None = None) -> List[float]:
        """Return timeseries list for *dataset* ('train' | 'valid')."""
        metric = metric or self._train_metric_name
        if isinstance(self.history, TrainingHistory):
            store = self.history.train if dataset == "train" else self.history.valid
            return store.get(metric, [])
        # Dict path – accept nested or flat
        if dataset in self.history:
            return self.history[dataset].get(metric, [])  # type: ignore[index]
        key = f"{dataset}_{metric}"
        return self.history.get(key, [])

    def _preds_targets(self) -> tuple[np.ndarray, np.ndarray]:
        if {"predictions", "targets"} <= self.model_outputs.keys():
            return (np.asarray(self.model_outputs["predictions"]),
                    np.asarray(self.model_outputs["targets"]))
        return (np.array([]), np.array([]))
    
    def _plot_forecasting_main(self) -> None:
        preds, targs = self._preds_targets()

        # If we have history, create the full 2x2 plot
        if self.has_history:
            train_loss = self._get_series("train")
            val_loss = self._get_series("valid")
            r2_vals = self._get_series("valid", "r2")

            plt.figure(figsize=(15, 10))
            # Subplot 1: Loss curves
            plt.subplot(2, 2, 1)
            plt.plot(train_loss, label="Train", lw=2)
            if val_loss:
                plt.plot(val_loss, label="Valid", lw=2)
            plt.xlabel("Epoch")
            plt.ylabel(self._train_metric_name.upper())
            plt.title("Loss curves")
            plt.legend()
            plt.grid(alpha=0.4)

            # Subplot 2: R² curve
            plt.subplot(2, 2, 2)
            if r2_vals:
                plt.plot(r2_vals, label="Valid R²", lw=2)
            if "r2" in self.metrics:
                plt.axhline(self.metrics["r2"], ls="--", color="red", label=f"Test R² = {self.metrics['r2']:.4f}")
            plt.title("R² over time")
            plt.xlabel("Epoch")
            plt.ylabel("R²")
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(alpha=0.4)
            
            # Subplot 3: Scatter actual vs predicted
            plt.subplot(2, 2, 3)

        # If no history, create a smaller 1x2 plot for just the evaluation results
        else:
            logger.info("No training history found. Generating evaluation-only plots.")
            plt.figure(figsize=(15, 5))
            # Subplot 1: Scatter actual vs predicted
            plt.subplot(1, 2, 1)
            
        # --- The rest of this code is shared by both layouts ---
        if preds.size and targs.size:
            plt.scatter(targs, preds, alpha=0.5, s=10)
            mn, mx = float(np.min([targs, preds])), float(np.max([targs, preds]))
            plt.plot([mn, mx], [mn, mx], "r--", lw=1)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs predicted")
        plt.grid(alpha=0.4)

        # Determine subplot position for the error distribution
        ax_pos = (2, 2, 4) if self.has_history else (1, 2, 2)
        plt.subplot(*ax_pos)
        
        if preds.size and targs.size:
            errors = preds - targs
            plt.hist(errors, bins=25, alpha=0.7)
            rmse = np.sqrt(np.mean(errors ** 2))
        else:
            rmse = float("nan")
        rmse_display = self.metrics.get("rmse", rmse)
        plt.title(f"Error distribution (RMSE = {rmse_display:.4f})")
        plt.xlabel("Prediction error")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.4)

        plt.tight_layout()
        out = self.output_dir / "forecasting_results.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved forecasting figure → %s", out)

    def _plot_forecasting_timeseries(self) -> None:
        preds, targs = self._preds_targets()
        if not preds.size or not targs.size:
            logger.warning("No predictions/targets available – skipping time‑series plot.")
            return
        n = min(150, len(targs))
        idx = np.arange(n)

        plt.figure(figsize=(14, 6))
        plt.plot(idx, targs[:n], label="Actual", marker="o", ms=3)
        plt.plot(idx, preds[:n], label="Predicted", marker="x", ms=3)
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.title(f"Predicted vs actual (first {n} samples)")
        plt.legend()
        plt.grid(alpha=0.4)
        out = self.output_dir / "forecasting_timeseries.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved forecasting time‑series → %s", out)
    
    def _save_forecasting_metrics(self) -> None:
        out = self.output_dir / "forecasting_metrics.txt"
        lines: list[str] = []
        
        # --- Overall metrics ---
        lines.append("Overall metrics:")
        lines.append(f"  MSE:   {self.metrics['mse']:.6f}")
        lines.append(f"  RMSE:  {self.metrics['rmse']:.6f}")
        lines.append(f"  MAE:   {self.metrics['mae']:.6f}")
        lines.append(f"  R²:    {self.metrics['r2']:.6f}")
        lines.append(f"  MAPE:  {self.metrics['mape']:.2f}%")
        lines.append("")
        
        # --- Per-horizon metrics ---
        if "per_horizon_metrics" in self.metrics:
            lines.append("Per-horizon metrics:")
            for h, vals in self.metrics["per_horizon_metrics"].items():
                vals_fmt = ", ".join(f"{k}={v:.4f}" for k, v in vals.items())
                lines.append(f"  {h}: {vals_fmt}")
            lines.append("")
        
        # --- Overall bin-conditioned metrics ---
        if "overall_bin_metrics" in self.metrics:
            lines.append("Overall bin-conditioned metrics:")
            # ADDED: Loop through the categories (Overall, Positive, Negative)
            for category_name, bins in self.metrics["overall_bin_metrics"].items():
                lines.append(f"  --- {category_name} Change Bins ---")
                for label, vals in bins.items():
                    vals_fmt = ", ".join(f"{k}={v:.4f}" for k, v in vals.items() if k != "n")
                    lines.append(f"    {label} (n={vals['n']}): {vals_fmt}")
            lines.append("")
        
        # --- Per-horizon bin-conditioned metrics ---
        if "per_horizon_bin_metrics" in self.metrics:
            lines.append("Per-horizon bin-conditioned metrics:")
            for h, h_bins in self.metrics["per_horizon_bin_metrics"].items():
                lines.append(f"  {h}:")
                # ADDED: Loop through the categories for each horizon
                for category_name, bins in h_bins.items():
                    lines.append(f"    --- {category_name} Change Bins ---")
                    for label, vals in bins.items():
                        vals_fmt = ", ".join(f"{k}={v:.4f}" for k, v in vals.items() if k != "n")
                        lines.append(f"      {label} (n={vals['n']}): {vals_fmt}")
            lines.append("")
        
        out.write_text("\n".join(lines))
        logger.info("Wrote forecasting metrics → %s", out)
    
    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _plot_classification(self) -> None:
        train_loss = self._get_series("train")
        val_loss = self._get_series("valid")

        # Predictions & labels
        labels = np.asarray(self.model_outputs.get("labels", []))
        probs = np.asarray(self.model_outputs.get("probabilities", []))
        preds = np.asarray(self.model_outputs.get("predictions", []))

        if "confusion_matrix" not in self.metrics and labels.size and preds.size:
            self.metrics["confusion_matrix"] = confusion_matrix(labels, preds)

        plt.figure(figsize=(14, 12))

        # Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(train_loss, label="Train", lw=2)
        if val_loss:
            plt.plot(val_loss, label="Valid", lw=2)
        plt.title("Loss curves")
        plt.xlabel("Epoch")
        plt.ylabel(self._train_metric_name.upper())
        plt.legend()
        plt.grid(alpha=0.4)

        # Confusion matrix
        plt.subplot(2, 2, 2)
        cm = self.metrics.get("confusion_matrix", np.zeros((2, 2), int))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Non‑Work", "Work"],
            yticklabels=["Non‑Work", "Work"],
        )
        plt.title("Confusion matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        # ROC curve
        plt.subplot(2, 2, 3)
        if labels.size and probs.size:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc_val = self.metrics["roc_auc"]
            plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc_val:.4f})")
        plt.plot([0, 1], [0, 1], ls="--", lw=1)
        plt.title("ROC curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.grid(alpha=0.4)

        # PR curve
        plt.subplot(2, 2, 4)
        if labels.size and probs.size:
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = self.metrics["auc_pr"]
            plt.plot(recall, precision, lw=2, label=f"PR (AP = {ap:.4f})")
        plt.title("Precision‑Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(alpha=0.4)

        plt.tight_layout()
        out = self.output_dir / "classification_results.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved classification figure → %s", out)

    def _save_classification_metrics(self) -> None:
        """Persist scalar classification metrics to a text‑file."""
        labels = np.asarray(self.model_outputs.get("labels", []))
        majority_acc = max(labels.mean(), 1 - labels.mean()) if labels.size else float("nan")
        acc = self.metrics["accuracy"]
        out = self.output_dir / "classification_metrics.txt"
        lines = [
            f"Accuracy:             {acc:.4f}",
            f"Balanced accuracy:    {self.metrics['balanced_accuracy']:.4f}",
            f"Precision:            {self.metrics['precision']:.4f}",
            f"Recall:               {self.metrics['recall']:.4f}",
            f"F1‑score:             {self.metrics['f1']:.4f}",
            f"ROC‑AUC:              {self.metrics['roc_auc']:.4f}",
            f"Average precision:    {self.metrics['auc_pr']:.4f}",
            f"Decision threshold:   {self.metrics['threshold']:.4f}",
            "",  # newline
            f"Baseline (majority‑class) accuracy: {majority_acc:.4f}",
            f"Improvement over baseline:          {((acc - majority_acc) / majority_acc * 100):.2f}%",
        ]
        out.write_text("\n".join(lines) + "\n")
        logger.info("Wrote classification metrics → %s", out)

    # ------------------------------------------------------------------
    # Feature importance (for LGBM)
    # ------------------------------------------------------------------

    def _plot_feature_importance(self) -> None:  # noqa: C901 – keep flat but readable
        """Bar‑chart of feature importance (gain)."""
        try:
            importance_raw = self.model.get_feature_importance(importance_type="gain")

            if isinstance(importance_raw, pd.DataFrame):
                imp_df = importance_raw.rename(columns=str.lower)
                if {"feature", "importance"} <= set(imp_df.columns):
                    imp_df = imp_df[["feature", "importance"]]
                else:
                    raise ValueError("DataFrame must contain 'feature' & 'importance' columns")
            else:
                # Assume ndarray or list‑like from LightGBM Booster
                importance_arr = np.asarray(importance_raw)
                try:
                    features = self.model.feature_name()
                except AttributeError:
                    features = [f"f{i}" for i in range(len(importance_arr))]
                imp_df = pd.DataFrame({"feature": features, "importance": importance_arr})

            top = imp_df.nlargest(30, "importance", keep="all").sort_values("importance")

            fig, ax = plt.subplots(figsize=(10, 12))
            sns.barplot(
                data=top,
                y="feature",
                x="importance",
                orient="h",
                errorbar=None,  # suppress future deprecation warning
                ax=ax,
            )
            ax.set_title("Top‑30 feature importance (gain)")
            fig.tight_layout()

            out = self.output_dir / "lgbm_feature_importance.png"
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            logger.info("Saved feature‑importance plot → %s", out)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Could not create feature‑importance plot: %s", exc, exc_info=True)