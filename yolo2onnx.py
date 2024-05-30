import torch
from ultralytics import YOLO

# YOLOv8 모델 불러오기 (pretrained 모델 사용 예시)
model = YOLO('/home/nvidia/Downloads/best.pt')  # 'best.pt' 대신 학습된 모델 파일 경로를 입력하세요.

# 입력을 위한 더미 데이터 생성
dummy_input = torch.randn(1, 3, 640, 640)  # YOLOv8 입력 크기 (1, 3, 640, 640) 사용

# 모델을 평가 모드로 설정
model.eval()

# 모델을 ONNX 형식으로 변환 및 저장
torch.onnx.export(
    model,                  # 학습된 모델
    dummy_input,            # 더미 입력 데이터
    "yolov8_model.onnx",    # 저장할 ONNX 파일 이름
    opset_version=11,       # ONNX opset 버전
    input_names=['input'],  # 입력 텐서 이름
    output_names=['output'],# 출력 텐서 이름
    dynamic_axes={
        'input': {0: 'batch_size'},  # 가변 배치 크기 지원
        'output': {0: 'batch_size'}
    }
)

print("ONNX 모델 변환 완료!")
