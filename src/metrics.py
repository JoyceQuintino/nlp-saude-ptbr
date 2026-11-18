"""Métricas e matriz de confusão compartilhadas pelos três modelos."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate(y_true, y_pred, label_names):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0
        ),
    }


def save_confusion_matrix(y_true, y_pred, label_names, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)), label_names, rotation=45, ha="right")
    ax.set_yticks(range(len(label_names)), label_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
