# FOD Detection 
---
## Introduction
---
FOD(Foreign Object Debris) Detection: 군 공항내에 위치하는 활주로 파편인 FOD를 검출하여 제거한다. 
Camera와 Radar를 모두 이용하여 2*2*2(cm)의 작은 크기의 물체를 검출하는 것이 목적이다.

#### 조건
+ Real-Time Object Detection
  + 0.3ms의 fast-inference 속도
+ 30km/h로 달리는 차량 위에서 물체 검출 
   + Blurring, Noise 고려
+ 2cm * 2cm * 2cm 이상의 물체 검출 : Tiny-Object Detection
   + Class 갯수: 최소 20개 이상 (FOD, Crack, Pothole)
 
## Environment
---
1. 보드: NVIDIA Jetson Orin AGX
2. 차량: FOD 검출 자율주행 차량 (사내 자체 제작 로봇)


### Lightweighting Methods
---
1.
2. 
