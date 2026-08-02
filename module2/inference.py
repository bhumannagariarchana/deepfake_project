import os
import torch
import numpy as np

from .deepfake_model import get_model
from .preprocessing import preprocess_aligned_face

# Import the FFT/Laplacian texture fallback we implemented earlier to ensure absolute resilience
# and to act as a fallback/auxiliary detector if model weights are missing.
from .deepfake_model_fallback import DeepfakeDetectorModel as FallbackTextureDetector

class DeepfakeInference:
    def __init__(self, model_path="models/deepfake_detector.pth"):
        self.model_path = model_path
        
        # Setup Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model = get_model(pretrained=False)
        self.has_weights = False

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model = self.model.to(self.device)
                self.model.eval()
                self.has_weights = True
                print(f"✓ EfficientNet-B0 detector initialized with weights: {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading EfficientNet-B0 weights: {str(e)}. Using fallback detector.")
        else:
            print(f"⚠️ Model weights not found at {model_path}. Fallback texture-based liveness detector is active.")
            
        # Instantiate fallback texture analyzer (FFT + Laplacian)
        self.fallback_detector = FallbackTextureDetector()

    def predict(self, image_path_or_array):
        """
        Runs inference on the aligned face image.
        Returns:
            real_prob (float): Probability that the image is a genuine live capture.
            fake_prob (float): Probability that the image is a deepfake/replay.
        """
        if self.has_weights:
            try:
                # 1. Preprocess aligned face image to standard PyTorch tensor
                tensor = preprocess_aligned_face(image_path_or_array)
                tensor = tensor.to(self.device)

                # 2. Run inference
                with torch.no_grad():
                    logits = self.model(tensor)
                    fake_prob = float(torch.sigmoid(logits).item())
                    real_prob = float(1.0 - fake_prob)
                    return real_prob, fake_prob
            except Exception as e:
                print(f"Inference error using EfficientNet-B0: {str(e)}. Redirecting to fallback...")
                
        # 3. Fallback: Run FFT/Laplacian texture analysis directly
        # Read image array if path is provided
        import cv2
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
        else:
            img = image_path_or_array

        real_prob = self.fallback_detector.predict(img)
        fake_prob = 1.0 - real_prob
        return real_prob, fake_prob
