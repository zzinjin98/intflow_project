import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot

def frame_extraction (text_url, video_url):
    cap = cv2.VideoCapture(video_url)
    #재생할 파일의 넓이 얻기
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #재생할 파일의 높이 얻기
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #재생할 파일의 프레임 레이트 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
    out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

    df = pd.read_csv(text_url, sep=",", header=None)
    frame = 0
    
    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==frame,1:].values.tolist()

                color = 255
                for i in df_frame:
                    points = rotate_box_dot(i[0], i[1], i[2], i[3], i[4])
                    img = cv2.polylines(img,[points],True,(color,color,color),thickness=3)
                    img = cv2.putText(img, str(int(i[11])), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_PLAIN, 2, (color,color,color), thickness=2)
                    
                out.write(img)
                cv2.imshow(video_url, img)
                cv2.waitKey(25)

                frame+=1
            else:
                break
                   
    else:
        print("can't open video")
        #재생할 파일의 넓이와 높이
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    frame_extraction("..\mounting_000\mounting_000_trk.txt", "..\mounting_000\mounting_000.mp4")
