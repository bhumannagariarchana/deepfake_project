import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # We use MediaPipe Face Mesh because it gives detailed landmarks
        # which we use for face alignment and liveness detection in one pass.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_face(self, frame):
        """
        Processes a frame to detect a face.
        Returns:
            face_detected (bool): True if face found.
            bbox (tuple): (x, y, w, h) in pixels, or None.
            landmarks (list): List of 478 landmarks (with refined irises), or None.
        """
        if frame is None:
            return False, None, None

        h, w, _ = frame.shape
        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return False, None, None

        # Take the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Calculate bounding box from landmarks
        x_coords = [lm.x for lm in face_landmarks.landmark]
        y_coords = [lm.y for lm in face_landmarks.landmark]

        # Convert to pixel coordinates
        x_min_px = int(min(x_coords) * w)
        x_max_px = int(max(x_coords) * w)
        y_min_px = int(min(y_coords) * h)
        y_max_px = int(max(y_coords) * h)

        # Clamp values to image dimensions
        x_min_px = max(0, x_min_px)
        y_min_px = max(0, y_min_px)
        x_max_px = min(w, x_max_px)
        y_max_px = min(h, y_max_px)

        bbox_w = x_max_px - x_min_px
        bbox_h = y_max_px - y_min_px
        bbox = (x_min_px, y_min_px, bbox_w, bbox_h)

        return True, bbox, face_landmarks.landmark

    def close(self):
        self.face_mesh.close()
