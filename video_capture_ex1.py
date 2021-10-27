import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot
import yaml

def frame_extraction (text_path, video_path):
    cap = cv2.VideoCapture(video_path)

    df = pd.read_csv(text_path, sep=",", header=None)
    frame = 0
    
    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==str(frame),1:].values.tolist()

                red_color = (0,0,255)
                green_color = (0,255,0)
                blue_color = (255,0,0)
                purple_color = (255,0,255)

                for i in df_frame:
                    
                    
                    points = rotate_box_dot(float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4]))
                    
                    # 전체
                    img = cv2.polylines(img,[points],True,red_color,thickness=3)

                    # 코 
                    img = cv2.circle(img,(int(float(i[5])),int(float(i[6]))), 5,green_color, thickness=3)

                    # 목
                    img = cv2.circle(img,(int(float(i[7])),int(float(i[8]))), 5,blue_color, thickness=3)

                    # 꼬리
                    img = cv2.circle(img,(int(float(i[9])),int(float(i[10]))), 5,purple_color, thickness=3)


                cv2.imshow(video_path, img)
                cv2.waitKey(50)

                frame+=1
            else:
                break

    else:
        print("can't open video")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    frame_extraction(config['det_path'],config['mp4_path'])