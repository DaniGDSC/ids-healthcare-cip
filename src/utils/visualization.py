"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap=cmap, values_format='.2f' if normalize else 'd')
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_roc_curves(y_true, y_proba, classes):
    n_classes = len(classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        RocCurveDisplay.from_predictions(y_true == classes[i], y_proba[:, i], ax=ax, name=f"ROC {classes[i]}")
    ax.set_title('ROC Curves')
    fig.tight_layout()
    return fig, ax