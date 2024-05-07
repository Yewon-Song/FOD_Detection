from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/home/skyauto/dataset/crack_yolov8/data.yaml', epochs=100, imgsz=640, amp=False)