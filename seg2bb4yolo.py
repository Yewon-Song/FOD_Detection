import cv2
import numpy as np
import os

def bbox2yolo(x_min, x_max, y_min, y_max, img_width, img_height):
        height = y_max - y_min
        width = x_max - x_min
        x_center  = x_min + (width/2)
        y_center = y_min + (height/2)

        yolo1 = x_center / img_width
        yolo2 = y_center / img_height
        yolo3 = width / img_width
        yolo4 = height / img_height
        return(yolo1, yolo2, yolo3, yolo4)

for n in range(240): # 여기서 240, {n:04d}는 변경대상임, 데이터에 따라 달라짐
        seg_path = f"/home/skyauto/dataset/pothole600/training/label/{n:04d}.png" #im.shape(400,400)
        save_path = f"/home/skyauto/dataset/diy_crackpot/train/labels/pothole600_train_labels/pothole600_train_{n:04d}.txt"

        im = cv2.imread(str(seg_path), 0)
        seg_value = 255

        # for i in os.listdir(seg_path):
        if im is not None:
                np_seg = np.array(im)
                segmentation = np.where(np_seg == seg_value) # segmentation은 seg의 해당 인덱스 값을 의미함

        bbox = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0: # 행과 열에 단 하나라도 0이 아닌 것이 있으면
                x_min = int(np.min(segmentation[1])) # 가장 왼쪽에 있는 열을 찾음
                x_max = int(np.max(segmentation[1])) # 가장 오른쪽에 있는 열을 찾음
                y_max = int(np.max(segmentation[0])) # 가장 위쪽에 있는 행을 찾음
                y_min = int(np.min(segmentation[0]))
                
                bbox = x_min, y_min, x_max, y_max
                print(bbox)

        print(bbox2yolo(x_min, y_min, x_max, y_max, 400, 400))

        data = bbox2yolo(x_min, y_min, x_max, y_max, 400, 400)
        data_str = ' '.join(str(x) for x in data)

        with open(save_path, 'w') as file:
                file.writelines(str('0 ' + data_str))
