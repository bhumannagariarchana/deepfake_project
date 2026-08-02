import os
import json
import matplotlib
matplotlib.use('Agg') # Thread-safe headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_training_plots(history, save_dir):
    """
    Saves the training loss and accuracy curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Plot Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], 'r-o', label='Training Loss')
    plt.plot(epochs, history["val_loss"], 'b-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Plot Accuracy Curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], 'r-o', label='Training Accuracy')
    plt.plot(epochs, history["val_acc"], 'b-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

def save_evaluation_plots(y_true, y_probs, cm, save_dir):
    """
    Generates and saves ROC Curve and Confusion Matrix heatmaps.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Plot Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Plot ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

def save_json(data, file_path):
    """Writes a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
