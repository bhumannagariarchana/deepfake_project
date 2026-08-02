import os
import sys
import json
import base64
import datetime
import time
import numpy as np
import cv2

from flask import Flask, render_template, request, jsonify, send_from_directory

# Make sure deepfake_project is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module1.login import validate_login, create_session
from module1.face_detection import FaceDetector
from module1.face_alignment import align_and_crop_face
from module1.challenge_generator import generate_challenge_sequence
from module1.liveness_detection import LivenessAnalyzer
from module1.challenge_verifier import ChallengeVerifier
from module1.metrics import MetricsTracker
from module2.authentication import authenticate_session

app = Flask(__name__)

# Directory paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Global trackers
face_detector = FaceDetector()
liveness_analyzer = LivenessAnalyzer(calibration_frames=30)
metrics_tracker = MetricsTracker()
active_verifier = None
active_session = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    global active_session, active_verifier
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    success, user_id, err_msg = validate_login(username, password)
    if not success:
        return jsonify({"success": False, "message": err_msg}), 401

    # Start Session
    active_session = create_session(user_id)
    
    # Generate 3 random challenges
    challenges = generate_challenge_sequence(length=3)
    active_session["challenge_sequence"] = challenges
    
    # Initialize verifier
    active_verifier = ChallengeVerifier(challenges, consecutive_required=5)
    liveness_analyzer.reset()

    return jsonify({
        "success": True,
        "user_id": user_id,
        "session_id": active_session["session_id"],
        "challenge_sequence": challenges
    })

@app.route("/reset_verification", methods=["POST"])
def reset_verification():
    liveness_analyzer.reset()
    if active_verifier:
        active_verifier.reset_action_state()
    return jsonify({"success": True})

@app.route("/process_frame", methods=["POST"])
def process_frame():
    global active_verifier, active_session
    data = request.json or {}
    image_b64 = data.get("image", "")
    session_id = data.get("session_id", "")

    if not active_session or active_session["session_id"] != session_id:
        return jsonify({"face_detected": False, "error": "Invalid or expired session"}), 400

    # 1. Decode base64 frame
    try:
        header, encoded = image_b64.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"face_detected": False, "error": f"Base64 decoding failed: {str(e)}"}), 400

    if frame is None:
        return jsonify({"face_detected": False, "error": "Decoded frame is empty"}), 400

    h, w, _ = frame.shape

    # 2. Run Face Detection
    start_time = time.time()
    detected, bbox, landmarks = face_detector.detect_face(frame)
    if not detected:
        latency_ms = (time.time() - start_time) * 1000
        metrics_tracker.record_frame_latency(latency_ms)
        return jsonify({
            "face_detected": False,
            "calibrated": liveness_analyzer.calibrated,
            "calibration_frames_collected": len(liveness_analyzer.calib_ears),
            "latency_ms": round(latency_ms, 2)
        })

    # Convert landmarks object array to serializable dict list for UI overlay rendering
    landmarks_serialized = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]

    # 3. Process frame metrics & calibration
    metrics, calibrated = liveness_analyzer.process_frame(landmarks, w, h)

    verifier_status = None
    if calibrated and active_verifier:
        # 4. Verify Active Challenge Action
        verifier_status = active_verifier.update(metrics, liveness_analyzer.baselines)
        
        # If the challenge sequence was completed successfully
        if verifier_status["is_completed"] and not verifier_status["failed"]:
            # Aligned and crop the face using face mesh landmarks
            aligned_face = align_and_crop_face(frame, landmarks)
            
            if aligned_face is not None:
                # Save aligned face crop
                face_path = os.path.join(OUTPUTS_DIR, "aligned_face.jpg")
                cv2.imwrite(face_path, aligned_face)

                # Save session JSON output
                session_json = {
                    "user_live": True,
                    "face_detected": True,
                    "face_aligned": True,
                    "challenge_score": 1.0,
                    "challenge_passed": True,
                    "aligned_face_path": "outputs/aligned_face.jpg", # match user spec
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                with open(os.path.join(OUTPUTS_DIR, "session.json"), "w") as f:
                    json.dump(session_json, f, indent=4)
                    
                active_session["status"] = "liveness_passed"

    latency_ms = (time.time() - start_time) * 1000
    metrics_tracker.record_frame_latency(latency_ms)

    return jsonify({
        "face_detected": True,
        "bbox": bbox,
        "landmarks": landmarks_serialized,
        "calibrated": calibrated,
        "calibration_frames_collected": len(liveness_analyzer.calib_ears),
        "verifier_status": verifier_status,
        "latency_ms": round(latency_ms, 2)
    })

@app.route("/authenticate", methods=["POST"])
def authenticate():
    global active_session
    data = request.json or {}
    session_id = data.get("session_id", "")

    if not active_session or active_session["session_id"] != session_id:
        return jsonify({"error": "Invalid session"}), 400

    # Execute Module 2 Deepfake Authentication Pipeline
    auth_result = authenticate_session(OUTPUTS_DIR)

    # Record liveness and deepfake metrics
    liveness_passed = auth_result.get("liveness_status") is not None and auth_result["liveness_status"].get("challenge_passed", False)
    deepfake_passed = auth_result.get("deepfake_status") is not None and auth_result["deepfake_status"].get("prediction") == "REAL"
    metrics_tracker.record_session_result(liveness_passed, deepfake_passed)

    # Load session output to return it
    session_json_path = os.path.join(OUTPUTS_DIR, "session.json")
    session_json = {}
    if os.path.exists(session_json_path):
        with open(session_json_path, "r") as f:
            session_json = json.load(f)

    return jsonify({
        "auth_result": auth_result,
        "session_json": session_json,
        "metrics": metrics_tracker.get_metrics_summary()
    })

@app.route("/submit_ground_truth", methods=["POST"])
def submit_ground_truth():
    data = request.json or {}
    system_decision_real = data.get("system_decision_real", False)
    ground_truth_real = data.get("ground_truth_real", False)
    
    metrics_tracker.record_ground_truth(system_decision_real, ground_truth_real)
    return jsonify({
        "success": True, 
        "metrics": metrics_tracker.get_metrics_summary()
    })

@app.route("/metrics", methods=["GET"])
def get_metrics_api():
    return jsonify(metrics_tracker.get_metrics_summary())

@app.route("/module1/outputs/<path:filename>")
@app.route("/outputs/<path:filename>")
def serve_output_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

def run_flask_app():
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)

if __name__ == "__main__":
    run_flask_app()
