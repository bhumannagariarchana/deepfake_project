from .deepfake_model import DeepfakeDetectorModel

class DeepfakeClassifier:
    def __init__(self, threshold=0.5):
        self.model = DeepfakeDetectorModel()
        self.threshold = threshold

    def classify(self, face_image):
        """
        Classifies whether the face is REAL or DEEPFAKE.
        Returns:
            is_real (bool): True if classified as real, False if deepfake.
            score (float): Reliability/probability score of real.
        """
        score = self.model.predict(face_image)
        is_real = score >= self.threshold
        return is_real, score
