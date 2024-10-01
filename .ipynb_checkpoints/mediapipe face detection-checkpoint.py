import cv2
import mediapipe as mp

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
            break

        # BGR 이미지를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 랜드마크 검출
        results = face_mesh.process(rgb_frame)

        # BGR 이미지로 다시 변환 (이미지 위에 그리기 위해)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # 얼굴 랜드마크가 검출된 경우
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 각 랜드마크에 점 찍기
                for idx, landmark in enumerate(face_landmarks.landmark):
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)

                    # 얼굴에 점 그리기 (파란색 점, 반지름 2)
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # 결과 프레임 출력
        cv2.imshow('Face Landmarks', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
