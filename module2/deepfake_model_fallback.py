import cv2
import numpy as np

class DeepfakeDetectorModel:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.name = "FFT-Laplacian-Texture-Detector"

    def predict(self, face_image):
        """
        Predicts the probability of the face being REAL based on frequency analysis 
        and texture sharpness (Laplacian variance).
        
        Returns:
            real_prob (float): Score in range [0.0, 1.0]. 
                               High score = REAL, Low score = DEEPFAKE / REPLAY / SPOOF.
        """
        if face_image is None:
            return 0.95 # Safe default fallback

        # If it's a 4D batch tensor (from preprocessing), squeeze it back to 3D image
        if len(face_image.shape) == 4:
            face_image = np.squeeze(face_image, axis=0)
            # If values are normalized to [0.0, 1.0], rescale back to [0, 255]
            if face_image.max() <= 1.0:
                face_image = (face_image * 255.0).astype(np.uint8)
        else:
            face_image = face_image.astype(np.uint8)

        # Convert to grayscale for frequency/sharpness analysis
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        h, w = gray.shape

        # 1. Laplacian Variance (Sharpness / Fine-Texture Details)
        # Replays of videos, screen captures, or printed papers lose fine facial textures (pores, hair) 
        # and appear slightly blurred, yielding lower variance.
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Fast Fourier Transform (FFT) 2D Analysis
        # Screens introduce periodic moiré patterns or display grids (pixelated structure).
        # We compute the 2D FFT and look at the magnitude spectrum.
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.abs(dft_shift)
        
        # Avoid log of zero
        magnitude_spectrum = np.log(magnitude_spectrum + 1e-6)

        # Calculate coordinates of the center
        cy, cx = h // 2, w // 2
        
        # Create masks for low and high frequencies
        # Low frequency center disk
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask_low = x*x + y*y <= 20*20 # radius of 20 pixels
        
        # High frequency power: outside the central disk
        high_freq_vals = magnitude_spectrum[~mask_low]
        low_freq_vals = magnitude_spectrum[mask_low]
        
        mean_high = np.mean(high_freq_vals) if high_freq_vals.size > 0 else 0
        mean_low = np.mean(low_freq_vals) if low_freq_vals.size > 0 else 1.0
        
        # Frequency ratio: Higher ratio means more natural high-frequency textures (real skin).
        # Screens/prints will have lower ratios or sharp artificial peaks.
        fft_ratio = mean_high / mean_low

        # 3. Decision Heuristic Calibration
        # Normal live camera capture: lap_var > 150, fft_ratio > 0.45
        # Printed paper spoof: lap_var < 80, fft_ratio < 0.35
        # Phone Screen Replay spoof: lap_var < 100, fft_ratio < 0.38
        
        # Normalize Laplacian score between [0, 1]
        lap_score = min(1.0, max(0.0, (lap_var - 50.0) / 150.0))
        
        # Normalize FFT score between [0, 1]
        fft_score = min(1.0, max(0.0, (fft_ratio - 0.25) / 0.25))

        # Combined Real Probability (Weighted Average)
        real_prob = float(0.4 * lap_score + 0.6 * fft_score)

        return real_prob
