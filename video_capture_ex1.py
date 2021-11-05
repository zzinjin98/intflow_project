import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot
import yaml
import random
import math
import os


def video_out(video_path,video_name):

    cap = cv2.VideoCapture(video_path) # VideoCapture 객체 정의
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # 코덱 정의

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)

    out = cv2.VideoWriter(video_name, fourcc, fps, (int(width), int(height))) # VideoWriter 객체 정의

    
    return out



def frame_extraction (config, PI = 3.14):
    text_path, video_path = config['trk_path'], config['mp4_path']

    out = video_out(video_path,os.path.splitext(os.path.basename(video_path))[0]+'.mp4')

    cap = cv2.VideoCapture(video_path)

    df = pd.read_csv(text_path, sep=",", header=None)
    frame = 0
    

    object_cnt = max(df[12]) + 1

    # 최대 개체 수만큼 랜덤으로 컬러를 만듦
    color_list = [(int(random.random()*i*256) % 25,int(random.random()*i*256) % 256,int(random.random()*i*256) % 256) for i in range(1,object_cnt+1)]

    color_nose = (255,255,255)
    color_neck = (255,0,0)
    color_tail = (255,255,0)

    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==frame,1:].values.tolist()
                

                for i in df_frame:
                    
                    
                    # 몸통 다각형
                    points = rotate_box_dot(i[0], i[1], i[2], i[3], i[4])
                    img = cv2.polylines(img,[points],True,color_list[int(i[11])],thickness=3)

                    # 몸통 타원 
                    # img = cv2.ellipse(img,((i[0],i[1]),(i[2],i[3]),float(i[4])*360/PI),color_list[int(i[11])],thickness=3)

                    # 코 
                    img = cv2.circle(img,(int(i[5]),int(i[6])), 5,color_nose, thickness=2)

                    # 목
                    img = cv2.circle(img,(int(i[7]),int(i[8])), 5,color_neck, thickness=2)

                    # 꼬리
                    img = cv2.circle(img,(int(i[9]),int(i[10])), 5,color_tail, thickness=2)
                    
                    # 코 목 선
                    cv2.line(img, (int(i[5]),int(i[6])), (int(i[7]), int(i[8])),color_list[int(i[11])],thickness=2)
                    length1 = round(math.dist([int(i[5]),int(i[6])],[int(i[7]),int(i[8])]),3)
                    img = cv2.putText(img, str(length1), (int((int(i[5])+int(i[7]))/2),int((int(i[6])+int(i[8]))/2)), cv2.FONT_HERSHEY_PLAIN, 1, color_list[int(i[11])], thickness=2)    

                    # 목 꼬리 선
                    cv2.line(img, (int(i[7]), int(i[8])), (int(i[9]),int(i[10])),color_list[int(i[11])],thickness=2)
                    length2 = round(math.dist([int(i[7]),int(i[8])],[int(i[9]),int(i[10])]),3)
                    img = cv2.putText(img, str(length2), (int((int(i[7])+int(i[9]))/2),int((int(i[8])+int(i[10]))/2)), cv2.FONT_HERSHEY_PLAIN, 1, color_list[int(i[11])], thickness=2) 

                    # 객체 번호
                    img = cv2.putText(img, str(int(i[11])), (int(i[0]),int(i[1])), cv2.FONT_HERSHEY_PLAIN, 2, color_list[int(i[11])], thickness=2)    

                    out.write(img)
                cv2.imshow(video_path, img)

                
                cv2.waitKey(2)

                frame+=1
            else:
                break

    else:
        print("can't open video")

    out.release()
    cap.release()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    frame_extraction(config)