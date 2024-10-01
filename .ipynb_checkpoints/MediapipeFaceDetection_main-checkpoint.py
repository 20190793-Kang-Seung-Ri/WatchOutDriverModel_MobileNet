import cv2
import mediapipe as mp
import numpy as np
import time
from DrowsinessFunction import calculate_ear
from DrowsinessFunction import calculate_perclos

# EAR 임계값 및 PERCLOS 계산을 위한 변수들
EAR_THRESHOLD = 0.15  # 눈이 감겼다고 판단할 임계값
BLINK_THRESHOLD = 30  # 깜박임 빈도 임계값
CLOSED_EYE_DURATION = 3  # 눈이 감긴 상태로 있는 시간(초)
FRAME_CHECK = 50  # 졸음 감지를 위한 프레임 체크 횟수
PERCLOS_THRESHOLD = 0.5  # PERCLOS 기준값 (50%)

# 변수 초기화
blink_count = 0
closed_eye_frames = 0
total_frames = 0
start_time = time.time()
blink_start_time = time.time()

# Mediapipe의 얼굴 메시 모듈과 그리기 유틸리티 설정
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 웹캠 연결
cap = cv2.VideoCapture(0)

# Mediapipe의 얼굴 랜드마크 검출기 생성
with mp_face_mesh.FaceMesh(
        max_num_faces=1,      # 검출할 얼굴 수
        refine_landmarks=True, # 더 세밀한 랜드마크 검출 활성화
        min_detection_confidence=0.5, # 얼굴 검출 신뢰도
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()  # 웹캠에서 프레임 읽기
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        # 프레임 좌우 반전
        frame = cv2.flip(frame, 1)

        # BGR 이미지를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 랜드마크 검출
        results = face_mesh.process(rgb_frame)

        # BGR 이미지로 다시 변환 (이미지 위에 그리기 위해)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        frame_height, frame_width, _ = frame.shape

        # 얼굴 랜드마크가 검출된 경우
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_indices = [33, 160, 158, 133, 153, 144]  # 왼쪽 눈
                right_eye_indices = [362, 385, 387, 263, 373, 380]  # 오른쪽 눈

                # 왼쪽 눈에 점 그리기
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]

                # 오른쪽 눈에 점 그리기
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]

                # EAR 계산
                left_ear = calculate_ear(left_eye_landmarks, frame_width, frame_height)
                right_ear = calculate_ear(right_eye_landmarks, frame_width, frame_height)
                avg_ear = (left_ear + right_ear) / 2.0

                # 눈이 감긴 상태인지 확인
                if avg_ear < EAR_THRESHOLD:
                    closed_eye_frames += 1
                else:
                    if closed_eye_frames > 0:
                        blink_count += 1  # 눈을 떴을 때 깜박임으로 기록
                    closed_eye_frames = 0

                total_frames += 1

                # PERCLOS 계산
                perclos = calculate_perclos(closed_eye_frames, total_frames)

                # 졸음 경고 기준
                if perclos > PERCLOS_THRESHOLD:
                    cv2.putText(frame, "Drowsiness Detected (PERCLOS)!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif avg_ear < EAR_THRESHOLD and closed_eye_frames > (CLOSED_EYE_DURATION * 30):  # 30 FPS 가정
                    cv2.putText(frame, "Drowsiness Detected (EAR)!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 1분 이내에 30번 이상 깜박임 확인
                if time.time() - blink_start_time < 60:  # 1분 이내
                    if blink_count >= BLINK_THRESHOLD:
                        cv2.putText(frame, "Too Many Blinks!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        blink_count = 0  # 카운트 초기화
                else:
                    # 1분이 지나면 타이머와 카운트 초기화
                    blink_start_time = time.time()
                    blink_count = 0

                # 화면에 표시
                cv2.putText(frame, f'EAR: {avg_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'PERCLOS: {perclos:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Blinks: {blink_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 결과 프레임 출력
        cv2.imshow('Eye Landmarks', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("프로그램 종료")
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
