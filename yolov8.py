from ultralytics import YOLO as yolo

model = yolo('yolov8n.pt')
#model.train(data='/home/skyauto/dataset/FOD/data.yaml ', epochs=100, amp=False)
# atch=2 imgsz=640 device=0 workers=8 optimizer=Adam pretrained=true val=true plots=true save=True show=true optimize=true lr0=0.001 lrf=0.01 fliplr=0.0 amp=False

# Segmentation for cracks and potholes
#model.train(data='/home/skyauto/dataset/diy_crackpot/data.yaml', epochs=100, amp=False, imgsz=640)

# voc dataset for FOD
model.train(data='/home/skyauto/dataset/FOD.v1-fod_v1.yolov8/data.yaml ', epochs=50, amp=False, imgsz=640)
