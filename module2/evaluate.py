import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from .dataset_loader import get_dataloaders
from .deepfake_model import get_model
from .utils import save_evaluation_plots, save_json

def evaluate_model(base_dir="datasets", model_path="models/deepfake_detector.pth", outputs_dir="outputs"):
    """
    Evaluates the trained EfficientNet-B0 model on the test dataset.
    Computes precision, recall, F1, AUC-ROC, FAR, FRR, and writes evaluation JSON and plots.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load test dataloader
    print("Loading Test Dataset...")
    _, _, test_loader = get_dataloaders(base_dir)

    # 2. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 3. Load Model
    model = get_model(pretrained=False)
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"⚠️ Warning: Model weights not found at {model_path}. Running evaluation with random weights.")
    
    model = model.to(device)
    model.eval()

    # 4. Evaluation Loop
    y_true = []
    y_pred = []
    y_probs = []

    print("Running inference on test dataset...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(float)
            
            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 5. Compute Metrics
    # In our mapping: 0 = Real, 1 = Fake
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except ValueError:
        # Fallback if only 1 class is present in test labels
        auc_score = 1.0

    # Confusion Matrix:
    # cm[0,0] = TN (Real as Real)
    # cm[0,1] = FP (Real as Fake) -> False Reject
    # cm[1,0] = FN (Fake as Real) -> False Accept
    # cm[1,1] = TP (Fake as Fake)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # FAR = False Acceptance Rate = fakes accepted as real = FN / (TP + FN)
    far = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # FRR = False Rejection Rate = reals rejected as fake = FP / (TN + FP)
    frr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(float(auc_score), 4),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "false_acceptance_rate": round(float(far), 4),
        "false_rejection_rate": round(float(frr), 4)
    }

    print("\n=== Evaluation Summary ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"FAR: {metrics['false_acceptance_rate']:.4f}")
    print(f"FRR: {metrics['false_rejection_rate']:.4f}")

    # 6. Save JSON and Plots
    metrics_path = os.path.join(outputs_dir, "model_evaluation.json")
    save_json(metrics, metrics_path)
    print(f"Evaluation metrics saved to: {metrics_path}")

    save_evaluation_plots(y_true, y_probs, cm, plots_dir)
    print(f"Evaluation visualization plots saved to: {plots_dir}")

    return metrics

if __name__ == "__main__":
    evaluate_model()
