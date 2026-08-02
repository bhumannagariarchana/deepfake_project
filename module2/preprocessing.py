import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_preprocessing_transforms(target_size=(224, 224)):
    """
    Returns standard torchvision preprocessing pipeline for evaluation.
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_augmentation_transforms(target_size=(224, 224)):
    """
    Returns data augmentation pipeline for training:
    - Random Horizontal Flip
    - Random Rotation
    - Color Jitter (Brightness/Contrast adjustment)
    - Normalization
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_aligned_face(image_path_or_array, target_size=(224, 224)):
    """
    Loads, standardizes, and normalizes an aligned face image for deepfake classification.
    
    Args:
        image_path_or_array (str or numpy.ndarray): Path to face crop image or BGR numpy array.
        target_size (tuple): Target image dimensions (width, height) for EfficientNet-B0.
        
    Returns:
        preprocessed_tensor (torch.Tensor): Normalised batch tensor [1, 3, 224, 224] ready for inference.
    """
    if isinstance(image_path_or_array, str):
        # Load image via OpenCV
        img_bgr = cv2.imread(image_path_or_array)
        if img_bgr is None:
            raise ValueError(f"Could not load image from path: {image_path_or_array}")
    else:
        img_bgr = image_path_or_array

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert NumPy array to PIL Image
    pil_img = Image.fromarray(img_rgb)

    # Apply standardization transforms
    transform = get_preprocessing_transforms(target_size)
    tensor = transform(pil_img)
    
    # Add batch dimension [1, 3, 224, 224]
    preprocessed_tensor = tensor.unsqueeze(0)
    
    return preprocessed_tensor
