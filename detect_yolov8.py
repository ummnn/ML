import cv2
from ultralytics import YOLO

# 비디오 로드
cap = cv2.VideoCapture('input_video.mp4')

# 학습된 YOLOv8 모델 로드
model = YOLO('best.pt')  # 학습된 모델 파일 경로

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지 수행
    results = model(frame)

    # 탐지 결과를 프레임에 그리기
    annotated_frame = results[0].plot()

    # 결과 비디오 출력
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
