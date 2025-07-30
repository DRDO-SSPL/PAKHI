# modules/utils.py

import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training History')
    plt.legend()
    plt.show()

def save_metrics(metrics_dict, path="output/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f)

def save_model(model, filename="output/model.h5"):
    os.makedirs("output", exist_ok=True)
    model.save(filename)

def save_sklearn_model(model, filename="output/sklearn_model.pkl"):
    """
    Save a scikit-learn model using joblib.
    """
    os.makedirs("output", exist_ok=True)
    joblib.dump(model, filename)

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()
