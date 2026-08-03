import os
import sys
import json
import time
import re
import urllib.request
import threading
import subprocess
import streamlit as st
from PIL import Image

# DEBUG: Try importing mediapipe and catch the exact exception
try:
    import mediapipe as mp
    print("DEBUG: Successfully imported mediapipe. Version:", mp.__version__)
    print("DEBUG: mediapipe path:", mp.__file__)
    # Try importing solutions explicitly
    import mediapipe.solutions.face_mesh as fm
    print("DEBUG: Successfully imported face_mesh.")
except Exception as e:
    import traceback
    print("DEBUG: MediaPipe Import failed with exception:")
    traceback.print_exc()


# Setup directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Background Flask Server Initialization
def run_flask_background():
    from app_flask import run_flask_app
    run_flask_app()

flask_thread_name = "FlaskBackgroundServer"
if not any(thread.name == flask_thread_name for thread in threading.enumerate()):
    t = threading.Thread(target=run_flask_background, name=flask_thread_name, daemon=True)
    t.start()
    time.sleep(1)

# Helper to fetch public IP of the host container/machine
def get_public_ip():
    try:
        return urllib.request.urlopen('https://ident.me', timeout=5).read().decode('utf8').strip()
    except Exception:
        try:
            return urllib.request.urlopen('https://ifconfig.me', timeout=5).read().decode('utf8').strip()
        except Exception:
            return "127.0.0.1"

# Fetch public IP on boot
if 'public_ip' not in st.session_state:
    st.session_state.public_ip = get_public_ip()

# 2. Background HTTP Tunnel Manager using localtunnel (bypasses cloud outbound SSH blocks)
tunnel_url_file = os.path.join(OUTPUTS_DIR, "tunnel_url.txt")

def run_http_tunnel_background():
    if os.path.exists(tunnel_url_file):
        try:
            os.remove(tunnel_url_file)
        except Exception:
            pass

    try:
        # Run localtunnel using npx (HTTP-based, not blocked by Streamlit firewalls)
        proc = subprocess.Popen(
            ["npx", "localtunnel", "--port", "5001"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in iter(proc.stdout.readline, ""):
            match = re.search(r"your url is:\s+(https?://\S+)", line)
            if match:
                url = match.group(1).strip().rstrip(".")
                with open(tunnel_url_file, "w") as f:
                    f.write(url)
                print(f"✓ Public HTTP Tunnel URL Active: {url}")
    except Exception as e:
        print(f"Failed to start HTTP tunnel: {e}")

tunnel_thread_name = "HTTPBackgroundTunnel"
if not any(thread.name == tunnel_thread_name for thread in threading.enumerate()):
    t_tunnel = threading.Thread(target=run_http_tunnel_background, name=tunnel_thread_name, daemon=True)
    t_tunnel.start()

# Streamlit Page Config
st.set_page_config(
    page_title="Deepfake Detection & Liveness Authentication",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header
st.markdown("<h1 style='text-align: center; font-family: Outfit; font-weight: 800; color: #10b981; margin-bottom: 0px;'>🛡️ SECUREAUTH</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af;'>Zero-Trust Liveness Verification & EfficientNet-B0 Deepfake Detection Suite</p>", unsafe_allow_html=True)
st.markdown("---")

# Navigation Selector
st.sidebar.title("System Control")
app_page = st.sidebar.radio("Go to:", [
    "🔐 Face Authentication Portal",
    "⚙️ AI Developer Dashboard"
])

def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return None
    return None

def get_iframe_url():
    if os.path.exists(tunnel_url_file):
        try:
            with open(tunnel_url_file, "r") as f:
                url = f.read().strip()
                if url.startswith("https://"):
                    return url
        except Exception:
            pass
    return "http://127.0.0.1:5001"

# ==========================================
# PAGE 1: FACE AUTHENTICATION PORTAL
# ==========================================
if app_page == "🔐 Face Authentication Portal":
    st.subheader("🔐 Biometric Verification Interface")
    st.markdown("Complete the user login and the randomized active liveness challenges.")

    iframe_url = get_iframe_url()
    
    # Render instructions if running on Streamlit Cloud
    if "loca.lt" in iframe_url:
        st.info(f"🔑 **Tunnel Endpoint IP Key**: `{st.session_state.public_ip}`")
        st.markdown(
            "*If the secure tunnel page below asks for a **Tunnel Password / Endpoint IP**, paste the key above and click **Click to Continue** to unlock your camera.*"
        )
    
    # Embed webcam liveness frame capture loop
    st.components.v1.iframe(iframe_url, height=700, scrolling=True)

    # Automatically check and render verification results below if available
    final_result = load_json_file(os.path.join(OUTPUTS_DIR, "final_result.json"))
    if final_result:
        st.markdown("### Verification Verdict")
        access = final_result.get("decision", {}).get("access", "DENIED")
        reason = final_result.get("decision", {}).get("reason", "")

        if access == "GRANTED":
            st.success(f"🔓 ACCESS GRANTED: {reason}")
        else:
            st.error(f"🔒 ACCESS DENIED: {reason}")

        col1, col2 = st.columns([1, 2])
        with col1:
            aligned_face_path = os.path.join(OUTPUTS_DIR, "aligned_face.jpg")
            if os.path.exists(aligned_face_path):
                st.image(Image.open(aligned_face_path), caption="Aligned Face Capture", use_column_width=True)
        with col2:
            st.write("#### Details")
            st.json(final_result)

# ==========================================
# PAGE 2: AI DEVELOPER DASHBOARD
# ==========================================
elif app_page == "⚙️ AI Developer Dashboard":
    st.subheader("⚙️ Model Training, Evaluation & Telemetry Dashboard")
    st.markdown("Manage model training runs, check performance metrics, and audit system telemetry logs.")

    # Section 1: Model Training Suite
    with st.expander("🏋️ EfficientNet-B0 Classifier Training", expanded=True):
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.write("##### Training Hyperparameters")
            epochs = st.slider("Epochs", min_value=1, max_value=30, value=3)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=0)
            learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.00001], index=1)
            
            if st.button("Start PyTorch Training Session"):
                st.info("Loading training splits...")
                with st.spinner("Training model backbone..."):
                    from module2.train import train_model
                    try:
                        train_model(
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            base_dir=os.path.join(PROJECT_DIR, "datasets"),
                            models_dir=MODELS_DIR,
                            outputs_dir=OUTPUTS_DIR
                        )
                        st.success("✓ Model checkpoint models/deepfake_detector.pth saved successfully!")
                    except Exception as e:
                        st.error(f"Training session error: {str(e)}")
        
        with col_t2:
            st.write("##### Performance Curves")
            loss_curve_path = os.path.join(PLOTS_DIR, "loss_curve.png")
            accuracy_curve_path = os.path.join(PLOTS_DIR, "accuracy_curve.png")
            
            if os.path.exists(loss_curve_path) and os.path.exists(accuracy_curve_path):
                st.image(Image.open(loss_curve_path), caption="Training vs Validation Loss", use_column_width=True)
                st.image(Image.open(accuracy_curve_path), caption="Training vs Validation Accuracy", use_column_width=True)
            else:
                st.warning("No performance curves found. Trigger a training run to render curves.")

    # Section 2: Model Evaluation Metrics
    with st.expander("📊 Evaluation Metrics & Visual Forensics", expanded=False):
        if st.button("Evaluate Classifier on Test Set"):
            with st.spinner("Generating confusion matrices..."):
                from module2.evaluate import evaluate_model
                try:
                    evaluate_model(
                        base_dir=os.path.join(PROJECT_DIR, "datasets"),
                        model_path=os.path.join(MODELS_DIR, "deepfake_detector.pth"),
                        outputs_dir=OUTPUTS_DIR
                    )
                    st.success("✓ Evaluation completed successfully!")
                except Exception as e:
                    st.error(f"Evaluation session error: {str(e)}")

        metrics = load_json_file(os.path.join(OUTPUTS_DIR, "model_evaluation.json"))
        if metrics:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            col_m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
            col_m3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
            col_m4.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")

            col_m5, col_m6, col_m7 = st.columns(3)
            col_m5.metric("AUC-ROC Score", f"{metrics['auc_roc']:.4f}")
            col_m6.metric("False Acceptance Rate (FAR)", f"{metrics['false_acceptance_rate'] * 100:.2f}%")
            col_m7.metric("False Rejection Rate (FRR)", f"{metrics['false_rejection_rate'] * 100:.2f}%")

            col_p1, col_p2 = st.columns(2)
            cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
            roc_plot_path = os.path.join(PLOTS_DIR, "roc_curve.png")
            
            with col_p1:
                if os.path.exists(cm_plot_path):
                    st.image(Image.open(cm_plot_path), caption="Confusion Matrix Heatmap", use_column_width=True)
            with col_p2:
                if os.path.exists(roc_plot_path):
                    st.image(Image.open(roc_plot_path), caption="Receiver Operating Characteristic (ROC)", use_column_width=True)
        else:
            st.warning("No evaluation metrics found. Please trigger 'Evaluate Classifier on Test Set'.")

    # Section 3: Telemetry Logs
    with st.expander("📝 Session JSON Logs", expanded=False):
        col_l1, col_l2 = st.columns(2)
        session_data = load_json_file(os.path.join(OUTPUTS_DIR, "session.json"))
        final_result = load_json_file(os.path.join(OUTPUTS_DIR, "final_result.json"))

        with col_l1:
            st.write("##### Module 1 Output: `session.json`")
            if session_data:
                st.json(session_data)
            else:
                st.warning("No liveness logs found.")
        with col_l2:
            st.write("##### Module 2 Output: `final_result.json`")
            if final_result:
                st.json(final_result)
            else:
                st.warning("No authentication logs found.")
