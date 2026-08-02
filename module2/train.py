import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe headless plotting
import matplotlib.pyplot as plt

from .dataset_loader import get_dataloaders
from .deepfake_model import get_model
from .utils import save_training_plots

def train_model(epochs=10, batch_size=16, learning_rate=0.0001, base_dir="datasets", models_dir="models", outputs_dir="outputs"):
    """
    Trains the EfficientNet-B0 model using Adam optimizer and BCEWithLogitsLoss.
    Implements learning rate scheduler, checkpointing, and early stopping.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "plots"), exist_ok=True)

    # 1. Load data loaders
    print("Initializing Data Loaders...")
    train_loader, val_loader, _ = get_dataloaders(base_dir, batch_size=batch_size)

    # 2. Setup Device (MPS for Mac M1/M2, CUDA for Nvidia, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using execution device: {device}")

    # 3. Instantiate model
    print("Loading pre-trained EfficientNet-B0 backbone...")
    model = get_model(pretrained=True).to(device)

    # 4. Optimizer, Loss & Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Early stopping and Checkpointing parameters
    patience = 5
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    checkpoint_path = os.path.join(models_dir, "deepfake_detector.pth")

    # History lists for plotting
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        # --- Training Step ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train_samples += inputs.size(0)

        epoch_train_loss = train_loss / total_train_samples
        epoch_train_acc = train_correct / total_train_samples

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                total_val_samples += inputs.size(0)

        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = val_correct / total_val_samples

        # Learning Rate adjustment based on validation loss
        scheduler.step(epoch_val_loss)

        # Record metrics history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")

        # Checkpointing (save based on best validation accuracy)
        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            print(f"✓ Validation accuracy improved ({best_val_acc:.4f} -> {epoch_val_acc:.4f}). Saving model...")
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Save final curves plots
    save_training_plots(history, os.path.join(outputs_dir, "plots"))
    
    # Save raw history data to json
    history_path = os.path.join(outputs_dir, "training_history.json")
    import json
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    return history

if __name__ == "__main__":
    train_model(epochs=3, batch_size=8)
