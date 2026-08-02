import cv2
import numpy as np

def align_and_crop_face(frame, landmarks, target_size=(256, 256), padding_ratio=0.3):
    """
    Aligns the face based on eye landmarks, rotates it to be upright, crops it, 
    and resizes it to the target size.
    
    Args:
        frame (numpy.ndarray): The original BGR input frame.
        landmarks (list): List of normalized MediaPipe landmarks.
        target_size (tuple): Output image dimensions (width, height).
        padding_ratio (float): Padding around the face bounding box relative to face size.
        
    Returns:
        aligned_face (numpy.ndarray): Aligned, cropped, and resized face image.
    """
    if frame is None or landmarks is None:
        return None

    h, w, _ = frame.shape

    # 1. Extract screen-left eye (subject's right) and screen-right eye (subject's left)
    # Landmarks 33 (outer corner) and 133 (inner corner) for screen-left eye
    # Landmarks 263 (outer corner) and 362 (inner corner) for screen-right eye
    x_33, y_33 = landmarks[33].x * w, landmarks[33].y * h
    x_133, y_133 = landmarks[133].x * w, landmarks[133].y * h
    x_263, y_263 = landmarks[263].x * w, landmarks[263].y * h
    x_362, y_362 = landmarks[362].x * w, landmarks[362].y * h

    left_eye_center = np.array([(x_33 + x_133) / 2.0, (y_33 + y_133) / 2.0])
    right_eye_center = np.array([(x_263 + x_362) / 2.0, (y_263 + y_362) / 2.0])

    # 2. Calculate the rotation angle (angle of eyes slope)
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(dY, dX))

    # Eye midpoint will be the center of rotation
    eye_midpoint = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                    (left_eye_center[1] + right_eye_center[1]) / 2.0)

    # 3. Get the rotation matrix and rotate the frame
    # In OpenCV, positive angle rotates counterclockwise
    M = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, M, (w, h))

    # 4. Rotate landmarks to find the bounding box in the rotated frame
    rotated_landmarks = []
    for lm in landmarks:
        px = lm.x * w
        py = lm.y * h
        # Apply the rotation matrix M
        rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        rotated_landmarks.append((rx, ry))

    rotated_landmarks = np.array(rotated_landmarks)

    # Calculate bounding box of the face in the rotated image
    rx_min, ry_min = np.min(rotated_landmarks, axis=0)
    rx_max, ry_max = np.max(rotated_landmarks, axis=0)

    # Face dimensions
    face_w = rx_max - rx_min
    face_h = ry_max - ry_min

    # Add padding around the bounding box
    pad_x = int(face_w * padding_ratio)
    pad_y = int(face_h * padding_ratio)

    # Crop coordinates with padding
    x_start = int(max(0, rx_min - pad_x))
    y_start = int(max(0, ry_min - pad_y))
    x_end = int(min(w, rx_max + pad_x))
    y_end = int(min(h, ry_max + pad_y))

    # 5. Crop and resize
    cropped_face = rotated_frame[y_start:y_end, x_start:x_end]
    
    if cropped_face.size == 0:
        # Fallback if crop is invalid
        return cv2.resize(frame, target_size)

    aligned_face = cv2.resize(cropped_face, target_size)
    return aligned_face
