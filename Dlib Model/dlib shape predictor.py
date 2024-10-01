import dlib
import cv2

# 얼굴 탐지를 위한 dlib의 HOG 기반 얼굴 탐지기
detector = dlib.get_frontal_face_detector()

# 얼굴 랜드마크 예측기 (68개의 얼굴 랜드마크를 예측하는 모델 경로)
predictor = dlib.shape_predictor('C:/_AppleBanana/WatchOutDriverModel/dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# 웹캠 연결 (기본적으로 첫 번째 카메라 사용)
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 grayscale로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 탐지
    faces = detector(gray)

    # 각 얼굴에 대해 랜드마크 예측
    for face in faces:
        # 얼굴 영역에 대한 랜드마크 추출
        landmarks = predictor(gray, face)
        
        # 랜드마크 68개에 대해 점찍기
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # 결과 프레임 보여주기
    cv2.imshow('Landmarks', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
