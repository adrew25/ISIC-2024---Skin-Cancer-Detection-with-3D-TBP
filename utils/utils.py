import torch
import numpy as np
from sklearn.metrics import roc_curve, auc


def calculate_partial_auc_by_tpr(y_true, y_scores, max_tpr=0.8):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= max_tpr)[0]
    if len(idx) == 0:
        return 0.0
    idx = idx[0]
    partial_auc = auc(fpr[: idx + 1], tpr[: idx + 1])
    partial_auc /= max_tpr
    return partial_auc


def save_best_model(model, epoch, partial_auc, best_auc, save_path):
    if partial_auc >= 0.8 and partial_auc > best_auc:
        best_auc = partial_auc
        torch.save(
            model.state_dict(),
            f"{save_path}_epoch_{epoch + 1}_auc_{partial_auc:.4f}.pth",
        )
        print(f"Model saved with partial AUC: {partial_auc:.4f}")
    return best_auc
