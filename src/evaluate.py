"""
Metrics for CMU-MOSEI evaluation (standard for MSA papers).

Regression:
  - MAE  (lower is better)
  - Corr (Pearson correlation, higher is better)
  - Acc7 (7-class accuracy, discretising [-3,3] into [-3,-2,-1,0,1,2,3])
  - Acc2 (binary: positive vs. non-positive, score > 0)
  - F1   (binary F1)

Classification (3-class):
  - Acc3  (3-class accuracy)
  - F1_w  (weighted F1)
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(preds: np.ndarray, gts: np.ndarray, task: str) -> dict:
    if task == "regression":
        return _regression_metrics(preds, gts)
    else:
        return _classification_metrics(preds.astype(int), gts.astype(int))


def _regression_metrics(preds, gts):
    mae = float(np.mean(np.abs(preds - gts)))

    # Pearson correlation
    if preds.std() < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(preds, gts)[0, 1])

    # 7-class accuracy: round to nearest integer in [-3, 3]
    preds_7 = np.clip(np.round(preds).astype(int), -3, 3)
    gts_7   = np.clip(np.round(gts).astype(int),   -3, 3)
    acc7 = float(accuracy_score(gts_7, preds_7))

    # Binary accuracy (positive = score > 0)
    preds_bin = (preds > 0).astype(int)
    gts_bin   = (gts   > 0).astype(int)
    acc2  = float(accuracy_score(gts_bin, preds_bin))
    f1_bin = float(f1_score(gts_bin, preds_bin, average="weighted", zero_division=0))

    return {"MAE": mae, "Corr": corr, "Acc7": acc7, "Acc2": acc2, "F1": f1_bin}


def _classification_metrics(preds, gts):
    acc3 = float(accuracy_score(gts, preds))
    f1_w = float(f1_score(gts, preds, average="weighted", zero_division=0))
    return {"Acc3": acc3, "F1_w": f1_w}
