from ultralytics import YOLO
import cv2
import math 

video = "/home/skyauto/fod/detection/FOD2_cam.mp4" # 협각렌즈: 1280*756

save =  "/home/skyauto/fod/crack"
cap = cv2.VideoCapture(video)
cap.set(3, 1280) #프레임너비: 640
cap.set(4, 756) #프레임높이: 380

global_min_area = float('inf')
global_min_box = None
global_frame_index = -1
frame_count = 0
model = YOLO("/home/skyauto/fod/detection/runs/detect/train/weights/best.pt")
classNames = ["FOD"]
set
x=10; y=200; w=1270; h=200
min_area = int()

while True:
    success, img = cap.read()
    roi = img[y:y+h, x:x+w]
    results = model(roi, stream=True)
    frame_min_area = float('inf')
    frame_min_box = None 
    img_copy = roi.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
 
            area = (x2-x1)*(y2-y1)

            if area < frame_min_area:
                frame_min_area = area
                frame_min_box = (x1, y1, x2, y2, classNames[int(box.cls[0])])

            cv2.rectangle(roi, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence -->",confidence)
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 0)
            thickness = 2
        if frame_min_area < global_min_area:
            global_min_area = frame_min_area
            global_min_box = frame_min_box
            global_frame_index = frame_count
    

        frame_count += 1

        cv2.putText(roi, classNames[cls], org, font, fontScale, color, thickness)
        cv2.imshow('Webcam', roi)
    if cv2.waitKey(1) == ord('q'):
        break
    
    if global_min_box:
        print(f"Global minimum area: {global_min_area}, Box: {global_min_box}, in frame {global_frame_index}")
cap.release()
cv2.destroyAllWindows()
