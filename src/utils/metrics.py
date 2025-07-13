from __future__ import annotations
from typing import Sequence, Dict, Any
import numpy as np
# Forecasting metrics
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
# Classificaation metrics
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, precision_recall_curve,
                             roc_auc_score, average_precision_score, balanced_accuracy_score)

# ---------- FORECASTING ----------
def regression_results(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    
    nonzero = y_true != 0
    mape = (np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])).mean() * 100 \
           if nonzero.any() else 0.0
    
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, mape=mape)

# ---------- CLASSIFICATION ----------
def binary_classification_results(
    y_true: Sequence[int], 
    proba: Sequence[float], 
    threshold: float = 0.5
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    proba  = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)

    # if only one class present → AUC is undefined; use 0.5
    auc  = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else 0.5
    aucp = average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else 0.5

    return dict(
        accuracy            = accuracy_score(y_true, y_pred),
        balanced_accuracy   = balanced_accuracy_score(y_true, y_pred),
        precision           = precision_score(y_true, y_pred, zero_division=0),
        recall              = recall_score(y_true, y_pred, zero_division=0),
        f1                  = f1_score(y_true, y_pred, zero_division=0),
        roc_auc             = auc,
        auc_pr              = aucp,
        confusion_matrix    = confusion_matrix(y_true, y_pred),
        threshold           = threshold,
    )

def find_optimal_f1_threshold(y_true, proba) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = np.where(
        precisions[:-1] + recalls[:-1] > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    return thresholds[np.argmax(f1_scores)] if thresholds.size else 0.5