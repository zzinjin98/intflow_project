import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot

df = pd.read_csv("mounting_001\mounting_001_det.txt", sep=",", header=None)
df.columns = ['frame','xc','yc','width','height','theta','no_x','no_y','ne_x','ne_y','ta_x','ta_y']
df = df.drop(index=0, axis=0)


rotate_list = []

for i in range(len(df)):
    data = list(df.loc[i+1,:])
    result = rotate_box_dot(int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]))
    rotate_list.append(result)

df_rotate = pd.DataFrame(rotate_list, columns=['frame','rotated_x1','rotated_y1', 'rotated_x2','rotated_y2', 'rotated_x3','rotated_y3', 'rotated_x4','rotated_y4'])


video_file = "mounting_001\mounting_001.mp4"

cap = cv2.VideoCapture(video_file)
frame = 0
if cap.isOpened():
    while True:
        # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장됨
        ret, img = cap.read()
        
        if ret:
            
            df_0 = df_rotate[df_rotate['frame']==frame]
            df_array_0 = []

            df_0.index = [i for i in range(len(df_0))]

            for i in range(len(df_0)):
                
                df_array_0.append(np.reshape(list(df_0.loc[i,:])[1:], (4,2)))

            color = 255
            for point in df_array_0 :
                
                img = cv2.polylines(img,[point],True,(color,color,color),thickness=3)
            # 1. 현재 frame 인 모든 정보를 가져와 리스트에 넣음
            # 2. 반복문을 통해 리스트의 정보 각각을 rotate_function으로 계산
            # 3. 계산된 좌표를 좌표 리스트에 저장
            # 4. 현재 frame의 img에 좌표를 찍어 다각형을 그림
            # point1 = np.array([[253,520],[264,918],[418,913],[407,516]], np.int32)



            cv2.imshow(video_file, img)
            cv2.waitKey(25)

            frame+=1
        else:
            break


else:
    print("can't open video")

cap.release()
cv2.destroyAllWindows()