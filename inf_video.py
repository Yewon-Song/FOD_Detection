from ultralytics import YOLO
import cv2
import math 
video = "/home/nvidia/Downloads/FOD2 (online-video-cutter.com).mp4"


cap = cv2.VideoCapture(video) 
cap.set(3, 640) #프레임너비: 640
cap.set(4, 380) #프레임높이: 380
model = YOLO("/home/nvidia/Downloads/tinybest.pt")

classNames = ["FOD"]
set
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence -->",confidence)
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()









