import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
import os


def calculate_partial_auc_by_tpr(y_true, y_scores, max_tpr=0.8):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= max_tpr)[0]
    if len(idx) == 0:
        return 0.0
    idx = idx[0]
    partial_auc = auc(fpr[: idx + 1], tpr[: idx + 1])
    partial_auc /= max_tpr
    return partial_auc


# From elefan
def pauc_sklearn(true, pred, min_tpr: float = 0.80) -> float:
    v_gt = abs(np.asarray(true) - 1)
    v_pred = np.array([1.0 - x for x in pred])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)

    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (
        partial_auc_scaled - 0.5
    )

    return partial_auc


# From elefan
def pauc(true, pred, min_tpr: float = 0.80) -> float:
    v_gt = abs(np.asarray(true) - 1)
    v_pred = -1.0 * np.asarray(pred)
    max_fpr = abs(1 - min_tpr)
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)

    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return partial_auc


def save_best_model(model, model_name, fold, epoch, val_loss, partial_auc, save_dir):
    model_file = f"{model_name}_fold{fold}_epoch{epoch}_valLoss{val_loss:.4f}_partialAUC{partial_auc:.4f}.pth"
    model_path = os.path.join(save_dir, model_file)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return val_loss, partial_auc
