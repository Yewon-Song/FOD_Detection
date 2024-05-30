import math
import cv2
from ultralytics import YOLO

# 이미지 경로 설정
image_path = "/home/nvidia/Downloads/FOD2 (online-video-cutter.com).mp4" # 협각렌즈 1280*756

# 이미지 읽기
img = cv2.imread(image_path)

# YOLO 모델 로드
model = YOLO("/home/nvidia/Downloads/best.pt")

# 클래스 이름 설정
classNames = ["FOD"]

# YOLO 모델을 통해 이미지 처리
results = model(img, stream=True)

# 객체 인식 결과 처리 및 시각화
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        confidence = math.ceil((box.conf[0]*100))/100
        print("Confidence -->", confidence)
        cls = int(box.cls[0])
        print("Class name -->", classNames[cls])
        org = (x1, y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 255, 0)
        thickness = 2
        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

# 결과 이미지 표시
cv2.imshow('Image', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
