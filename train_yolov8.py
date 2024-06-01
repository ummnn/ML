from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # YOLOv8 Nano 모델 파일 사용

# 모델 학습
model.train(data='data.yaml', epochs=50, imgsz=640)