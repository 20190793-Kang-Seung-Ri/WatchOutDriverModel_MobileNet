import cv2
import numpy as np
import time
import tensorflow as tf

# 눈 감김 예측 모델 로드
model = tf.keras.models.load_model('path_to_your_eye_blink_model.h5')

# 임계값 및 PERCLOS 관련 변수
PERCLOS_THRESHOLD = 0.5  # PERCLOS 기준값 (50%)
CLOSED_EYE_DURATION = 3  # 눈이 감긴 상태로 있는 시간(초)
BLINK_THRESHOLD = 30  # 깜박임 빈도 임계값
closed_eye_frames = 0
total_frames = 0
blink_count = 0
blink_start_time = time.time()

# 웹캠 연결
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        break

    # 프레임 좌우 반전
    frame = cv2.flip(frame, 1)
    resized_frame = cv2.resize(frame, (224, 224))  # 모델 입력 크기로 조정

    # 프레임 정규화 및 차원 확장 (모델 입력에 맞게 변환)
    input_frame = np.expand_dims(resized_frame / 255.0, axis=0)

    # 눈 감김 예측 (1이면 눈 감김, 0이면 눈 뜸)
    prediction = model.predict(input_frame)
    eye_state = np.argmax(prediction, axis=1)[0]

    # 눈 감김 상태 확인
    if eye_state == 1:  # 눈 감김 상태로 판단된 경우
        closed_eye_frames += 1
    else:
        if closed_eye_frames > 0:
            blink_count += 1  # 눈을 뜬 경우 깜박임으로 기록
        closed_eye_frames = 0

    total_frames += 1

    # PERCLOS 계산
    perclos = closed_eye_frames / total_frames

    # 졸음 경고 기준
    if perclos > PERCLOS_THRESHOLD:
        cv2.putText(frame, "Drowsiness Detected (PERCLOS)!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif closed_eye_frames > (CLOSED_EYE_DURATION * 30):  # 30 FPS 가정
        cv2.putText(frame, "Drowsiness Detected (Closed Eyes)!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 1분 내 깜박임 빈도 확인
    if time.time() - blink_start_time < 60:  # 1분 이내
        if blink_count >= BLINK_THRESHOLD:
            cv2.putText(frame, "Too Many Blinks!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            blink_count = 0  # 카운트 초기화
    else:
        # 1분이 지나면 타이머와 카운트 초기화
        blink_start_time = time.time()
        blink_count = 0

    # 화면에 표시
    cv2.putText(frame, f'PERCLOS: {perclos:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Blinks: {blink_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 프레임 출력
    cv2.imshow('Drowsiness Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("프로그램 종료")
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
