import numpy as np

# EAR(Eye Aspect Ratio) 계산 함수
def calculate_ear(eye_landmarks, frame_width, frame_height):
    # 수직 거리
    A = np.linalg.norm(np.array([eye_landmarks[1].x * frame_width, eye_landmarks[1].y * frame_height]) -
                       np.array([eye_landmarks[5].x * frame_width, eye_landmarks[5].y * frame_height]))
    B = np.linalg.norm(np.array([eye_landmarks[2].x * frame_width, eye_landmarks[2].y * frame_height]) -
                       np.array([eye_landmarks[4].x * frame_width, eye_landmarks[4].y * frame_height]))
    # 수평 거리
    C = np.linalg.norm(np.array([eye_landmarks[0].x * frame_width, eye_landmarks[0].y * frame_height]) -
                       np.array([eye_landmarks[3].x * frame_width, eye_landmarks[3].y * frame_height]))
    ear = (A + B) / (2.0 * C)
    return ear

# PERCLOS 측정
def calculate_perclos(closed_eyes_frames, total_frames):
    return closed_eyes_frames / total_frames if total_frames > 0 else 0
