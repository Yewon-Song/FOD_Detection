import cv2

# 동영상 파일 경로
video_path = '/home/skyauto/fod/detection/drivingTest (online-video-cutter.com).mp4'

# 동영상 파일 불러오기
cap = cv2.VideoCapture(video_path)

# 동영상이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 동영상 재생
while True:
    # 프레임별로 동영상 읽기
    ret, frame = cap.read()
    
    # 더 이상 읽을 프레임이 없으면 반복문 탈출
    if not ret:
        print("Reached the end of the video.")
        break

    # 읽은 프레임을 화면에 표시
    cv2.imshow('Video Playback', frame)
    
    # 'q' 키를 누르면 재생 중지
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()