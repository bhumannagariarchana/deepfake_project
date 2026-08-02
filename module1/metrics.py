import numpy as np

class MetricsTracker:
    def __init__(self):
        # Latency tracking
        self.frame_latencies_ms = []
        
        # Session ground truth statistics
        # Decision: "REAL" (Auth success) vs "SPOOF" (Auth failure - Liveness or Deepfake failed)
        self.stats = {
            "TP": 0,  # Decided REAL, Ground Truth was REAL
            "FP": 0,  # Decided REAL, Ground Truth was SPOOF (False Acceptance)
            "TN": 0,  # Decided SPOOF, Ground Truth was SPOOF
            "FN": 0,  # Decided SPOOF, Ground Truth was REAL (False Rejection)
            "total_liveness_pass": 0,
            "total_liveness_fail": 0,
            "total_deepfake_pass": 0,
            "total_deepfake_fail": 0,
            "total_sessions": 0
        }

    def record_frame_latency(self, latency_ms):
        self.frame_latencies_ms.append(latency_ms)
        # Keep only last 1000 frames to prevent memory leaks
        if len(self.frame_latencies_ms) > 1000:
            self.frame_latencies_ms.pop(0)

    def record_session_result(self, liveness_passed, deepfake_passed):
        self.stats["total_sessions"] += 1
        if liveness_passed:
            self.stats["total_liveness_pass"] += 1
        else:
            self.stats["total_liveness_fail"] += 1

        if liveness_passed and deepfake_passed:
            self.stats["total_deepfake_pass"] += 1
        elif liveness_passed:
            self.stats["total_deepfake_fail"] += 1

    def record_ground_truth(self, system_decision_real, ground_truth_real):
        """
        Records the user-submitted ground truth for accuracy/FAR/FRR.
        system_decision_real (bool): True if system authenticated the user, False otherwise.
        ground_truth_real (bool): True if the actual person was a live real human, False if spoof.
        """
        if system_decision_real:
            if ground_truth_real:
                self.stats["TP"] += 1
            else:
                self.stats["FP"] += 1
        else:
            if ground_truth_real:
                self.stats["FN"] += 1
            else:
                self.stats["TN"] += 1

    def get_metrics_summary(self):
        avg_latency = float(np.mean(self.frame_latencies_ms)) if self.frame_latencies_ms else 0.0
        
        tp = self.stats["TP"]
        fp = self.stats["FP"]
        tn = self.stats["TN"]
        fn = self.stats["FN"]
        
        total_evals = tp + fp + tn + fn
        
        accuracy = (tp + tn) / total_evals if total_evals > 0 else 1.0
        
        # FAR = FP / (FP + TN)
        far_denom = fp + tn
        far = fp / far_denom if far_denom > 0 else 0.0
        
        # FRR = FN / (TP + FN)
        frr_denom = tp + fn
        frr = fn / frr_denom if frr_denom > 0 else 0.0
        
        return {
            "avg_latency_ms": round(avg_latency, 2),
            "accuracy": round(accuracy, 4),
            "far": round(far, 4),
            "frr": round(frr, 4),
            "total_evaluations": total_evals,
            "confusion_matrix": {
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn
            },
            "stats": self.stats.copy()
        }
