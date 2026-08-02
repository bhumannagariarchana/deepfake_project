import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocessing import get_preprocessing_transforms, get_augmentation_transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom PyTorch Dataset for loading face images.
        Expected directory structure:
            root_dir/real/
            root_dir/fake/
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Classes mapping
        self.class_to_idx = {"real": 0, "fake": 1}

        # Gather real samples
        real_dir = os.path.join(root_dir, "real")
        if os.path.exists(real_dir):
            for file_path in glob.glob(os.path.join(real_dir, "*.*")):
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.samples.append((file_path, 0))

        # Gather fake/deepfake samples
        fake_dir = os.path.join(root_dir, "fake")
        if os.path.exists(fake_dir):
            for file_path in glob.glob(os.path.join(fake_dir, "*.*")):
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.samples.append((file_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load image via OpenCV
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            # Settle on a random empty image if reading fails
            img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        if self.transform:
            tensor = self.transform(pil_img)
        else:
            # Fallback transforms
            transform = get_preprocessing_transforms()
            tensor = transform(pil_img)

        return tensor, torch.tensor(label, dtype=torch.float32)

def generate_mock_datasets(base_dir="datasets", num_samples=64):
    """
    Creates dummy directories and writes solid color/noise images to simulate
    real and deepfake splits for zero-config validation training.
    """
    splits = ["train", "validation", "test"]
    categories = ["real", "fake"]

    for split in splits:
        for cat in categories:
            dir_path = os.path.join(base_dir, split, cat)
            os.makedirs(dir_path, exist_ok=True)
            
            # Check if directory already has files
            existing_files = glob.glob(os.path.join(dir_path, "*.*"))
            if len(existing_files) >= 5:
                continue # Skip if already populated

            # Write mock images
            for i in range(num_samples):
                # Real images: smooth gradients simulating faces
                # Fake images: grids or noise overlays simulating deepfakes
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                if cat == "real":
                    # Smooth circle
                    cv2.circle(img, (112, 112), 60, (120, 180, 100), -1)
                else:
                    # Alternating grids (simulating moire/deepfake artifacts)
                    for x in range(0, 224, 8):
                        cv2.line(img, (x, 0), (x, 224), (100, 50, 150), 1)
                
                # Add some Gaussian noise
                noise = np.random.normal(0, 15, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                file_name = f"mock_{split}_{cat}_{i}.jpg"
                cv2.imwrite(os.path.join(dir_path, file_name), img)

def get_dataloaders(base_dir="datasets", batch_size=32, target_size=(224, 224)):
    """
    Loads train, validation, and test datasets and returns DataLoader objects.
    Automatically generates mock data if folders are missing.
    """
    # 1. Ensure mock datasets are created if directories are empty
    generate_mock_datasets(base_dir)

    # 2. Get augmentation transforms for training and standard transforms for validation/testing
    train_transform = get_augmentation_transforms(target_size)
    val_transform = get_preprocessing_transforms(target_size)

    # 3. Instantiate datasets
    train_dataset = DeepfakeDataset(os.path.join(base_dir, "train"), transform=train_transform)
    val_dataset = DeepfakeDataset(os.path.join(base_dir, "validation"), transform=val_transform)
    test_dataset = DeepfakeDataset(os.path.join(base_dir, "test"), transform=val_transform)

    # 4. Instantiate dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
