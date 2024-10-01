import cv2
import numpy as np
import tensorflow as tf

# 얼굴 랜드마크 예측 모델 로드 (양 눈, 코, 양쪽 입)
landmark_model = tf.keras.models.load_model('path_to_your_landmark_model.h5')

# 머리 기울기 각도 임계값 설정
HEAD_TILT_THRESHOLD = 20  # 기울기 각도 임계값

# 머리 기울기를 계산하는 함수
def calculate_head_tilt(landmarks):
    # 좌표로 양 눈과 코, 입 양쪽 좌표 추출
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # 눈의 중심점 계산
    eye_center = (left_eye + right_eye) / 2

    # 입의 중심점 계산
    mouth_center = (left_mouth + right_mouth) / 2

    # 기울기 계산 (눈 중심점과 입 중심점 사이의 기울기)
    delta_x = mouth_center[0] - eye_center[0]
    delta_y = mouth_center[1] - eye_center[1]
    
    # 각도 계산 (radians to degrees)
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    return angle

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

    # 얼굴 랜드마크 예측
    predictions = landmark_model.predict(input_frame)
    landmarks = predictions.reshape(5, 2)  # (x, y) 좌표 형태로 변환

    # 기울기 계산
    head_tilt_angle = calculate_head_tilt(landmarks)

    # 기울기 확인 및 경고 메시지
    if abs(head_tilt_angle) > HEAD_TILT_THRESHOLD:
        cv2.putText(frame, "Head Tilt Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 화면에 표시
    cv2.putText(frame, f'Head Tilt Angle: {head_tilt_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 프레임 출력
    cv2.imshow('Head Tilt Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("프로그램 종료")
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
