// App state
let sessionState = {
    userId: null,
    sessionId: null,
    challengeSequence: [],
    currentIdx: 0,
    isCalibrated: false,
    timer: 20,
    timerInterval: null,
    streamActive: false
};

// DOM Elements
const loginSection = document.getElementById('login-section');
const livenessSection = document.getElementById('liveness-section');
const resultSection = document.getElementById('result-section');

const loginForm = document.getElementById('login-form');
const usernameInput = document.getElementById('username');
const passwordInput = document.getElementById('password');

const webcam = document.getElementById('webcam');
const canvasOverlay = document.getElementById('canvas-overlay');
const ctx = canvasOverlay.getContext('2d');

const calibrationOverlay = document.getElementById('calibration-overlay');
const calibrationProgress = document.getElementById('calibration-progress');
const alertBanner = document.getElementById('alert-banner');
const alertText = document.getElementById('alert-text');

const activeChallengeText = document.getElementById('active-challenge-text');
const activeGestureProgress = document.getElementById('active-gesture-progress');

const sessionUserId = document.getElementById('session-user-id');
const sessionIdDisplay = document.getElementById('session-id-display');
const challengeList = document.getElementById('challenge-list');
const sessionTimerDisplay = document.getElementById('session-timer');

const successView = document.getElementById('success-view');
const failureView = document.getElementById('failure-view');
const alignedFaceImg = document.getElementById('aligned-face-img');
const authResultPre = document.getElementById('auth-result-pre');
const sessionJsonPre = document.getElementById('session-json-pre');
const failureTitle = document.getElementById('failure-title');
const failureDesc = document.getElementById('failure-desc');

let videoStream = null;
let frameSendInterval = null;

// Handle Username input change to toggle password field visibility for EMP IDs
usernameInput.addEventListener('input', (e) => {
    const val = e.target.value.trim();
    const isEmpId = /^EMP\d+$/i.test(val);
    const passwordGroup = document.getElementById('password-group');
    if (isEmpId) {
        passwordGroup.style.opacity = '0.4';
        passwordInput.required = false;
    } else {
        passwordGroup.style.opacity = '1';
        passwordInput.required = true;
    }
});

// Login Form Submit
loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = usernameInput.value.trim();
    const password = passwordInput.value;

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();

        if (response.ok && data.success) {
            // Success: Initialise Liveness session
            sessionState.userId = data.user_id;
            sessionState.sessionId = data.session_id;
            sessionState.challengeSequence = data.challenge_sequence;
            sessionState.currentIdx = 0;
            sessionState.isCalibrated = false;
            sessionState.timer = 20;

            // Update UI Sidebar
            sessionUserId.textContent = sessionState.userId;
            sessionIdDisplay.textContent = sessionState.sessionId.substring(0, 18) + '...';
            renderChallengeList();

            // Transition to Liveness View
            loginSection.classList.remove('active');
            livenessSection.classList.add('active');
            
            // Start Camera
            await startWebcam();
        } else {
            alert(data.message || 'Login failed.');
        }
    } catch (error) {
        console.error('Error during login:', error);
        alert('Server communication error.');
    }
});

// Start Webcam capture
async function startWebcam() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });
        webcam.srcObject = videoStream;
        sessionState.streamActive = true;

        // Sync overlay canvas resolution with video dimensions
        webcam.onloadedmetadata = () => {
            canvasOverlay.width = webcam.videoWidth;
            canvasOverlay.height = webcam.videoHeight;
            
            // Start the frame capture-send loop (~12 FPS)
            frameSendInterval = setInterval(sendFrameToServer, 85);
            
            // Reset backend state
            fetch('/reset_verification', { method: 'POST' });
        };
    } catch (err) {
        console.error('Webcam access error:', err);
        alert('Webcam access is required for liveness verification.');
        restartAuth();
    }
}

// Draw Face Overlays (Bounding Box and key landmarks)
function drawFaceOverlay(bbox, landmarks) {
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

    if (!bbox) return;

    const [x, y, w, h] = bbox;

    // 1. Draw Bounding Box (Glowing rounded rectangle)
    ctx.strokeStyle = '#10b981'; // emerald
    ctx.lineWidth = 3;
    ctx.shadowBlur = 15;
    ctx.shadowColor = 'rgba(16, 185, 129, 0.4)';
    
    // Custom drawing for rounded rect
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 12);
    ctx.stroke();

    // 2. Draw Landmark dots for feedback (eyes, brows, nose, mouth)
    ctx.fillStyle = '#0ea5e9'; // cyber blue
    ctx.shadowBlur = 0; // reset shadow
    
    // Select a subset of critical landmarks for HUD visual effect
    // Eyes: 33, 133, 159, 145 (left); 263, 362, 386, 374 (right)
    // Eyebrows: 70, 107, 285, 336
    // Nose outline: 4, 1, 10, 152
    // Lips outer: 61, 291, 13, 14
    const keyIndices = [
        33, 133, 159, 145, 263, 362, 386, 374, 
        70, 107, 285, 336, 
        4, 1, 10, 152, 
        61, 291, 13, 14
    ];

    keyIndices.forEach(idx => {
        if (landmarks[idx]) {
            const lm = landmarks[idx];
            const px = lm.x * canvasOverlay.width;
            const py = lm.y * canvasOverlay.height;
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

// Send Frame to Server for analysis
async function sendFrameToServer() {
    if (!sessionState.streamActive) return;

    // Create temp canvas to extract JPEG
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = webcam.videoWidth;
    tempCanvas.height = webcam.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(webcam, 0, 0, tempCanvas.width, tempCanvas.height);
    
    const base64Image = tempCanvas.toDataURL('image/jpeg', 0.85);

    try {
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                image: base64Image,
                session_id: sessionState.sessionId
            })
        });
        const data = await response.json();

        // 1. Handle alert/warnings (e.g. no face, off-center)
        if (!data.face_detected) {
            alertBanner.classList.remove('hidden');
            alertText.textContent = "No Face Detected";
            ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
            return;
        } else {
            alertBanner.classList.add('hidden');
        }

        // 2. Draw visual landmarks feedback
        drawFaceOverlay(data.bbox, data.landmarks);

        // 3. Calibration phase handling
        if (!data.calibrated) {
            calibrationOverlay.classList.add('active');
            const progressPct = (data.calibration_frames_collected / 30) * 100;
            calibrationProgress.style.width = `${progressPct}%`;
            return;
        } else {
            // Once calibrated, hide calibration overlay and start session timer countdown
            if (!sessionState.isCalibrated) {
                sessionState.isCalibrated = true;
                calibrationOverlay.classList.remove('active');
                startTimer();
            }
        }

        // 4. Update gesture progress bar & active instructions
        if (data.verifier_status) {
            const status = data.verifier_status;
            
            // Check index change to update list rendering
            if (status.current_idx !== sessionState.currentIdx) {
                sessionState.currentIdx = status.current_idx;
                renderChallengeList();
            }

            // Update gesture text
            if (status.current_action) {
                activeChallengeText.textContent = getActionDescription(status.current_action);
                activeGestureProgress.style.width = `${status.active_progress * 100}%`;
            }

            // Check if completed challenge sequence
            if (status.is_completed) {
                handleVerificationPass();
            }
        }

    } catch (err) {
        console.error('Frame processing error:', err);
    }
}

// Timer Countdown
function startTimer() {
    clearInterval(sessionState.timerInterval);
    sessionState.timer = 20;
    sessionTimerDisplay.textContent = `${sessionState.timer}s`;
    
    sessionState.timerInterval = setInterval(() => {
        sessionState.timer--;
        sessionTimerDisplay.textContent = `${sessionState.timer}s`;

        if (sessionState.timer <= 0) {
            handleVerificationFail("Session Timeout", "Liveness challenges were not completed within the 20-second time limit.");
        }
    }, 1000);
}

// Dynamic Action descriptive titles
function getActionDescription(action) {
    switch (action) {
        case "Smile": return "Please SMILE widely 🙂";
        case "Blink": return "Please BLINK once 😉";
        case "Blink twice": return "Please BLINK TWICE 😉😉";
        case "Turn Left": return "Please TURN your head LEFT 👤←";
        case "Turn Right": return "Please TURN your head RIGHT 👤→";
        case "Look Up": return "Please LOOK UP 👤↑";
        case "Look Down": return "Please LOOK DOWN 👤↓";
        case "Raise Eyebrows": return "Please RAISE your EYEBROWS 🤨";
        case "Open Mouth": return "Please OPEN your MOUTH 😮";
        case "Close Mouth": return "Please CLOSE your MOUTH 😐";
        default: return "Please look straight...";
    }
}

// Render challenge items sidebar list
function renderChallengeList() {
    challengeList.innerHTML = '';
    sessionState.challengeSequence.forEach((action, idx) => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'checklist-item';

        if (idx < sessionState.currentIdx) {
            itemDiv.classList.add('success');
            itemDiv.innerHTML = `<span class="check-icon">✓</span> <span class="check-label">${action}</span>`;
        } else if (idx === sessionState.currentIdx) {
            itemDiv.classList.add('active');
            itemDiv.innerHTML = `<span class="check-icon">●</span> <span class="check-label">${action}</span>`;
        } else {
            itemDiv.classList.add('pending');
            itemDiv.innerHTML = `<span class="check-icon">○</span> <span class="check-label">${action}</span>`;
        }
        challengeList.appendChild(itemDiv);
    });
}

// Stop Video Capture
function stopWebcam() {
    sessionState.streamActive = false;
    clearInterval(frameSendInterval);
    clearInterval(sessionState.timerInterval);

    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
}

// Handle Verification PASS -> Triggers Module 2 Authentication
async function handleVerificationPass() {
    stopWebcam();

    // Transition to Loading State in results section
    livenessSection.classList.remove('active');
    resultSection.classList.add('active');
    successView.classList.remove('hidden');
    failureView.classList.add('hidden');

    try {
        // Query server to run Module 2 Deepfake verification & authentication
        const response = await fetch('/authenticate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionState.sessionId })
        });
        const data = await response.json();

        // Render aligned face crop image
        alignedFaceImg.src = `/module1/outputs/aligned_face.jpg?t=${new Date().getTime()}`;

        // Populate Code Tabs
        authResultPre.querySelector('code').textContent = JSON.stringify(data.auth_result, null, 2);
        sessionJsonPre.querySelector('code').textContent = JSON.stringify(data.session_json, null, 2);

        // Update success metrics
        if (data.metrics) {
            updateMetricsDisplay(data.metrics);
        }

        // Reset feedback panel buttons state
        document.querySelectorAll('.feedback-panel').forEach(p => {
            p.style.opacity = '1';
            p.querySelectorAll('button').forEach(b => b.disabled = false);
        });

        // Highlight code syntax (if libraries like Prism were used, but raw text format is fine)
        if (!data.auth_result.authenticated) {
            // If liveness passed but deepfake failed
            handleVerificationFail("Access Denied", data.auth_result.reason, data.metrics);
        }

    } catch (err) {
        console.error("Auth payload retrieval error:", err);
        handleVerificationFail("Module 2 Error", "Liveness verification passed, but deepfake detection module failed to execute.");
    }
}

// Handle Verification FAIL
function handleVerificationFail(title, desc, metrics = null) {
    stopWebcam();

    livenessSection.classList.remove('active');
    resultSection.classList.add('active');
    successView.classList.add('hidden');
    failureView.classList.remove('hidden');

    failureTitle.textContent = title;
    failureDesc.textContent = desc;

    // Reset feedback panel buttons state
    document.querySelectorAll('.feedback-panel').forEach(p => {
        p.style.opacity = '1';
        p.querySelectorAll('button').forEach(b => b.disabled = false);
    });

    if (metrics) {
        updateMetricsDisplay(metrics);
    } else {
        fetchMetrics();
    }
}

// Tab Switch helper
window.switchTab = function(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content pre').forEach(pre => pre.classList.remove('active'));

    if (tabId === 'auth-json') {
        document.querySelector("[onclick=\"switchTab('auth-json')\"]").classList.add('active');
        authResultPre.classList.add('active');
    } else {
        document.querySelector("[onclick=\"switchTab('session-json')\"]").classList.add('active');
        sessionJsonPre.classList.add('active');
    }
};

// Restart verification
window.restartAuth = function() {
    stopWebcam();
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    
    resultSection.classList.remove('active');
    livenessSection.classList.remove('active');
    loginSection.classList.add('active');

    usernameInput.value = '';
    passwordInput.value = '';
    document.getElementById('password-group').style.opacity = '1';
};

// Render Metrics on the UI
function updateMetricsDisplay(metrics) {
    if (!metrics) return;
    const formatPct = (val) => (val * 100).toFixed(1) + "%";
    
    // Success panel metrics
    const latSuccess = document.getElementById('metric-latency-success');
    if (latSuccess) {
        latSuccess.textContent = metrics.avg_latency_ms + " ms";
        const accSuccess = document.getElementById('metric-accuracy-success');
        if (accSuccess) accSuccess.textContent = formatPct(metrics.accuracy);
        const farSuccess = document.getElementById('metric-far-success');
        if (farSuccess) farSuccess.textContent = formatPct(metrics.far);
        const frrSuccess = document.getElementById('metric-frr-success');
        if (frrSuccess) frrSuccess.textContent = formatPct(metrics.frr);
        const totalSuccess = document.getElementById('metric-total-success');
        if (totalSuccess) totalSuccess.textContent = metrics.total_evaluations;
    }

    // Failure panel metrics
    const latFailure = document.getElementById('metric-latency-failure');
    if (latFailure) {
        latFailure.textContent = metrics.avg_latency_ms + " ms";
        const accFailure = document.getElementById('metric-accuracy-failure');
        if (accFailure) accFailure.textContent = formatPct(metrics.accuracy);
        const farFailure = document.getElementById('metric-far-failure');
        if (farFailure) farFailure.textContent = formatPct(metrics.far);
        const frrFailure = document.getElementById('metric-frr-failure');
        if (frrFailure) frrFailure.textContent = formatPct(metrics.frr);
        const totalFailure = document.getElementById('metric-total-failure');
        if (totalFailure) totalFailure.textContent = metrics.total_evaluations;
    }
}

// Fetch metrics from API
async function fetchMetrics() {
    try {
        const response = await fetch('/metrics');
        const metrics = await response.json();
        updateMetricsDisplay(metrics);
    } catch (err) {
        console.error("Error fetching metrics:", err);
    }
}

// Submit Ground Truth Feedback
window.submitFeedback = async function(systemDecisionReal, groundTruthReal) {
    // Disable feedback buttons to prevent duplicate clicks
    document.querySelectorAll('.feedback-panel button').forEach(btn => btn.disabled = true);
    document.querySelectorAll('.feedback-panel').forEach(panel => panel.style.opacity = '0.5');

    try {
        const response = await fetch('/submit_ground_truth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                system_decision_real: systemDecisionReal,
                ground_truth_real: groundTruthReal
            })
        });
        const data = await response.json();
        
        if (data.success) {
            updateMetricsDisplay(data.metrics);
            alert("Feedback recorded! Experimental metrics updated.");
        }
    } catch (err) {
        console.error("Error submitting ground truth:", err);
        alert("Failed to submit ground truth.");
        document.querySelectorAll('.feedback-panel button').forEach(btn => btn.disabled = false);
        document.querySelectorAll('.feedback-panel').forEach(panel => panel.style.opacity = '1');
    }
};
