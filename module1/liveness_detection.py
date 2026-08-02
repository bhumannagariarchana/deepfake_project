import numpy as np

class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

def euclidean_dist(p1, p2, w, h):
    """
    Computes Euclidean distance between two landmarks in pixel space.
    """
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class LivenessAnalyzer:
    def __init__(self, calibration_frames=30):
        self.calibration_frames = calibration_frames
        self.calibrated = False
        
        # Lists to store calibration metrics
        self.calib_ears = []
        self.calib_mars = []
        self.calib_smiles = []
        self.calib_yaws = []
        self.calib_pitches = []
        self.calib_eyebrows = []
        
        # Calibrated baselines
        self.baselines = {
            "ear": 0.3,
            "mar": 0.05,
            "smile": 0.65,
            "yaw": 0.5,
            "pitch": 1.0,
            "eyebrow": 0.25
        }

    def reset(self):
        """Resets the analyzer for a new session."""
        self.calibrated = False
        self.calib_ears.clear()
        self.calib_mars.clear()
        self.calib_smiles.clear()
        self.calib_yaws.clear()
        self.calib_pitches.clear()
        self.calib_eyebrows.clear()

    def get_metrics(self, landmarks, w, h):
        """
        Extracts raw metrics from facial landmarks.
        """
        # --- Eye Aspect Ratio (EAR) ---
        # Screen-left eye (subject's right)
        left_p1 = landmarks[33]   # Outer corner
        left_p4 = landmarks[133]  # Inner corner
        left_p2 = landmarks[159]  # Top
        left_p6 = landmarks[145]  # Bottom
        left_p3 = landmarks[158]  # Top outer
        left_p5 = landmarks[153]  # Bottom outer
        
        left_vertical = euclidean_dist(left_p2, left_p6, w, h) + euclidean_dist(left_p3, left_p5, w, h)
        left_horizontal = euclidean_dist(left_p1, left_p4, w, h)
        left_ear = left_vertical / (2.0 * left_horizontal) if left_horizontal > 0 else 0

        # Screen-right eye (subject's left)
        right_p1 = landmarks[263]  # Outer corner
        right_p4 = landmarks[362]  # Inner corner
        right_p2 = landmarks[386]  # Top
        right_p6 = landmarks[374]  # Bottom
        right_p3 = landmarks[387]  # Top inner
        right_p5 = landmarks[373]  # Bottom inner

        right_vertical = euclidean_dist(right_p2, right_p6, w, h) + euclidean_dist(right_p3, right_p5, w, h)
        right_horizontal = euclidean_dist(right_p1, right_p4, w, h)
        right_ear = right_vertical / (2.0 * right_horizontal) if right_horizontal > 0 else 0

        mean_ear = (left_ear + right_ear) / 2.0

        # --- Mouth Aspect Ratio (MAR) ---
        m_left = landmarks[61]   # Corner
        m_right = landmarks[291] # Corner
        m_top = landmarks[13]    # Inner upper lip
        m_bottom = landmarks[14] # Inner lower lip

        m_vertical = euclidean_dist(m_top, m_bottom, w, h)
        m_horizontal = euclidean_dist(m_left, m_right, w, h)
        mar = m_vertical / m_horizontal if m_horizontal > 0 else 0

        # --- Smile Ratio ---
        # Normalize mouth width by the outer eye distance
        eye_distance = euclidean_dist(landmarks[33], landmarks[263], w, h)
        smile_ratio = m_horizontal / eye_distance if eye_distance > 0 else 0

        # --- Yaw Ratio (Left/Right Turn) ---
        # x-coordinate of nose tip relative to the cheeks
        # CHEEKS: Left 234, Right 454. NOSE TIP: 4
        # Since coordinates are normalised, we calculate:
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        nose_tip = landmarks[4]
        
        yaw_width = right_cheek.x - left_cheek.x
        yaw_ratio = (nose_tip.x - left_cheek.x) / yaw_width if yaw_width > 0 else 0.5

        # --- Pitch Ratio (Look Up/Down) ---
        # y-coordinate of nose tip relative to forehead (10) and chin (152)
        forehead = landmarks[10]
        chin = landmarks[152]
        
        upper_dist = nose_tip.y - forehead.y
        lower_dist = chin.y - nose_tip.y
        pitch_ratio = upper_dist / lower_dist if lower_dist > 0 else 1.0

        # --- Eyebrow Ratio (Raise Eyebrows) ---
        # Distance from eyes to eyebrows
        # Left eye center: midpoint between 33 and 133
        # Right eye center: midpoint between 263 and 362
        left_eye_center = Point(
            (landmarks[33].x + landmarks[133].x) / 2.0,
            (landmarks[33].y + landmarks[133].y) / 2.0
        )
        right_eye_center = Point(
            (landmarks[263].x + landmarks[362].x) / 2.0,
            (landmarks[263].y + landmarks[362].y) / 2.0
        )

        left_eyebrow = landmarks[70]   # Outer left eyebrow
        right_eyebrow = landmarks[285] # Outer right eyebrow

        left_eb_dist = euclidean_dist(left_eyebrow, left_eye_center, w, h)
        right_eb_dist = euclidean_dist(right_eyebrow, right_eye_center, w, h)
        mean_eyebrow_dist = (left_eb_dist + right_eb_dist) / 2.0
        
        eyebrow_ratio = mean_eyebrow_dist / eye_distance if eye_distance > 0 else 0.25

        return {
            "ear": mean_ear,
            "mar": mar,
            "smile": smile_ratio,
            "yaw": yaw_ratio,
            "pitch": pitch_ratio,
            "eyebrow": eyebrow_ratio
        }

    def process_frame(self, landmarks, w, h):
        """
        Processes a frame. If not calibrated, gathers calibration data.
        Returns metrics and whether calibration is complete.
        """
        metrics = self.get_metrics(landmarks, w, h)

        if not self.calibrated:
            self.calib_ears.append(metrics["ear"])
            self.calib_mars.append(metrics["mar"])
            self.calib_smiles.append(metrics["smile"])
            self.calib_yaws.append(metrics["yaw"])
            self.calib_pitches.append(metrics["pitch"])
            self.calib_eyebrows.append(metrics["eyebrow"])

            if len(self.calib_ears) >= self.calibration_frames:
                self.baselines["ear"] = float(np.median(self.calib_ears))
                self.baselines["mar"] = float(np.median(self.calib_mars))
                self.baselines["smile"] = float(np.median(self.calib_smiles))
                self.baselines["yaw"] = float(np.median(self.calib_yaws))
                self.baselines["pitch"] = float(np.median(self.calib_pitches))
                self.baselines["eyebrow"] = float(np.median(self.calib_eyebrows))
                self.calibrated = True

        return metrics, self.calibrated
