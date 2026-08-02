import os
import json
import datetime
from .inference import DeepfakeInference

def authenticate_session(module1_outputs_dir, deepfake_threshold=0.5):
    """
    Main authentication orchestrator (Module 2).
    Reads liveness outputs from Module 1, runs deepfake classification on the aligned face crop,
    decides access clearance, and saves outputs/final_result.json.
    """
    session_path = os.path.join(module1_outputs_dir, "session.json")
    aligned_face_path = os.path.join(module1_outputs_dir, "aligned_face.jpg")
    final_result_path = os.path.join(module1_outputs_dir, "final_result.json")

    # 1. Read liveness session metadata
    if not os.path.exists(session_path):
        result = {
            "authenticated": False,
            "decision": {
                "access": "DENIED",
                "reason": "Active Liveness Verification Failed: Session metadata not found."
            }
        }
        with open(final_result_path, "w") as f:
            json.dump(result, f, indent=4)
        return result

    with open(session_path, "r") as f:
        session_data = json.load(f)

    # 2. Check if liveness check passed in Module 1
    if not session_data.get("challenge_passed", False) or not session_data.get("user_live", False):
        result = {
            "authenticated": False,
            "liveness_status": {
                "challenge_passed": False,
                "challenge_score": session_data.get("challenge_score", 0.0)
            },
            "decision": {
                "access": "DENIED",
                "reason": "Active Liveness Verification Failed: Challenges were not passed."
            }
        }
        with open(final_result_path, "w") as f:
            json.dump(result, f, indent=4)
        return result

    # 3. Check if aligned face image is present
    if not os.path.exists(aligned_face_path):
        result = {
            "authenticated": False,
            "liveness_status": {
                "challenge_passed": True,
                "challenge_score": session_data.get("challenge_score", 1.0)
            },
            "decision": {
                "access": "DENIED",
                "reason": "Active Liveness Verification Passed, but Aligned face image not found."
            }
        }
        with open(final_result_path, "w") as f:
            json.dump(result, f, indent=4)
        return result

    try:
        # 4. Initialize Deepfake Inference
        # We specify models/deepfake_detector.pth
        model_weights_path = os.path.join(os.path.dirname(module1_outputs_dir), "models", "deepfake_detector.pth")
        
        inference = DeepfakeInference(model_path=model_weights_path)
        
        # 5. Run inference on aligned face
        real_prob, fake_prob = inference.predict(aligned_face_path)
        
        is_real = real_prob >= deepfake_threshold
        prediction = "REAL" if is_real else "DEEPFAKE"
        access = "GRANTED" if is_real else "DENIED"
        
        if is_real:
            reason = "Live user verified and face classified as genuine."
        else:
            reason = f"Authentication rejected. Face detected as potential deepfake/replay."

        result = {
            "authenticated": is_real,
            "liveness_status": {
                "challenge_passed": True,
                "challenge_score": session_data.get("challenge_score", 1.0)
            },
            "deepfake_status": {
                "model": "EfficientNet-B0",
                "prediction": prediction,
                "real_probability": round(real_prob, 4),
                "deepfake_probability": round(fake_prob, 4),
                "confidence": round(real_prob * 100 if is_real else fake_prob * 100, 2),
                "threshold": deepfake_threshold
            },
            "decision": {
                "access": access,
                "reason": reason
            }
        }

        # Save outputs/final_result.json
        with open(final_result_path, "w") as f:
            json.dump(result, f, indent=4)

        return result

    except Exception as e:
        result = {
            "authenticated": False,
            "decision": {
                "access": "DENIED",
                "reason": f"Error running deepfake classification model: {str(e)}"
            }
        }
        with open(final_result_path, "w") as f:
            json.dump(result, f, indent=4)
        return result
