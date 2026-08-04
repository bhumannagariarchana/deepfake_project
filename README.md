# Multi-Modal Liveness Verification & Deepfake Face Authentication System

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-10b981?logo=github&logoColor=white)](https://github.com/bhumannagariarchana/deepfake_project)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepfakeproject-xqdcfn8t9b8cspq7p83fsd.streamlit.app)

🚀 **Live Deployment URL**:[(https://deepfakeproject-xqdcfn8t9b8cspq7p83fsd.streamlit.app)](https://deepfakeproject-xqdcfn8t9b8cspq7p83fsd.streamlit.app)

This project implements a complete binary **Liveness Verification (Module 1)** and **AI-Based Deepfake Detection (Module 2)** system designed for secure face-based access control.

---

## 🏗️ Architecture

```
deepfake_project/
├── module1/                      # Module 1 Liveness Verification
│   ├── login.py                  # Credentials verification & session initiation
│   ├── face_detection.py         # MediaPipe Face Mesh wrapper for bounding boxes
│   ├── face_alignment.py         # Affine transformations, rotation, scaling & cropping
│   ├── challenge_generator.py    # Zero-trust challenge sequence builder (3-4 actions)
│   ├── liveness_detection.py     # Real-time feature calculation & user calibration
│   ├── challenge_verifier.py     # Check action metrics against calibrated baselines
│   └── metrics.py                # Frame latency tracking and ground-truth calculator
├── module2/                      # Module 2 Deepfake Detection
│   ├── dataset_loader.py         # Loads train/validation/test sets & mock dataset writer
│   ├── preprocessing.py          # Formats aligned face crop into EfficientNet tensors
│   ├── train.py                  # PyTorch model training loop (Adam, BCE, Early stopping)
│   ├── evaluate.py               # Evaluates test dataset (Accuracy, AUC, FAR, FRR)
│   ├── inference.py              # Runs inference on aligned crops with fallbacks
│   ├── deepfake_model.py         # EfficientNet-B0 binary classification network
│   ├── deepfake_model_fallback.py# OpenCV/NumPy texture-based anti-spoofing fallback
│   ├── authentication.py         # Combines Module 1 and 2 to write final_result.json
│   └── utils.py                  # Plotting curves, heatmaps, and JSON IO helpers
├── datasets/                     # Directory for FaceForensics++, Celeb-DF, and DFDC splits
├── models/
│   └── deepfake_detector.pth     # Saved best model state-dict weights
├── outputs/                      # Session and evaluation results directory
│   ├── aligned_face.jpg          # Cropped, aligned face from Module 1
│   ├── session.json              # Session liveness metadata
│   ├── final_result.json         # Integrated Module 1 & 2 final auth decisions
│   ├── model_evaluation.json     # Model test evaluation metrics
│   └── plots/                    # Loss, accuracy, confusion matrix, and ROC curves
├── templates/
│   └── index.html                # High-fidelity tracking webcam interface
├── static/                       # Web styles and JS capture loops
├── app.py                        # Streamlit visual dashboard
├── app_flask.py                  # Flask webcam frame processing background API
├── requirements.txt              # System package dependencies
└── README.md                     # Documentation
```

---

## 🚀 Get Started

### 1. Installation
Install the necessary python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application (Streamlit + Flask Hybrid)
Start the Streamlit dashboard in your terminal:
```bash
streamlit run app.py
```
This single command spins up:
1.  **Flask background thread** on port `5001` (to process base64 webcam frame streams at 12 FPS and run real-time landmark tracking).
2.  **Streamlit user interface** on port `8501`.

👉 Open your browser to: **[http://localhost:8501](http://localhost:8501)**

---

## 🛠️ Main Features & Modules

### 1. 🔐 Portal Verification
*   Complete the user login using demo credentials (e.g., `admin` / `admin123` or Employee ID `EMP101`).
*   Align your face for the 3-second calibration phase.
*   Complete the zero-trust randomized challenge checklist (Smile, Blink, Turn Left/Right, etc.).
*   Once cleared, **Module 2** automatically intercepts the aligned face crop (`outputs/aligned_face.jpg`) and runs the **EfficientNet-B0** classifier.
*   The final integrated decision is generated and saved under `outputs/final_result.json`.

### 2. 🏋️ Model Training Hub
*   Choose epochs, batch sizes, and learning rates.
*   Click **Start EfficientNet-B0 Model Training** to run the PyTorch training loop on organized dataset splits.
*   *Zero-Configuration*: If the training folders are empty, the dataset loader will automatically generate a mock texture dataset so that the training pipeline compiles and runs successfully out of the box.
*   Validation accuracy checkpoints are saved to `models/deepfake_detector.pth`.

### 3. 📊 Evaluation & Visualizations
*   Click **Re-evaluate Model on Test Split** to compute metrics against the testing split.
*   The system calculates: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **AUC-ROC**, **FAR**, and **FRR**, saved as `outputs/model_evaluation.json`.
*   Directly renders training curves, ROC curve, and Confusion Matrix heatmaps.
*   Interactive ground-truth buttons allow users to submit actual contexts to dynamically update biometric stats.
