import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot

def frame_extraction (text_url, video_url):
    cap = cv2.VideoCapture(video_url)

    df = pd.read_csv(text_url, sep=",", header=None)
    frame = 0
    
    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==str(frame),1:].values.tolist()

                color = 255
                for i in df_frame:
                    points = rotate_box_dot(float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4]))
                    img = cv2.polylines(img,[points],True,(color,color,color),thickness=3)

                cv2.imshow(video_url, img)
                cv2.waitKey(25)

                frame+=1
            else:
                break

    else:
        print("can't open video")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_extraction("..\mounting_000\mounting_000_det.txt", "..\mounting_000\mounting_000.mp4")

