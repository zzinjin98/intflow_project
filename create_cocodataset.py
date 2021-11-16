import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot
import yaml
import random
import math
import os
import json



def frame_extraction (config, PI = 3.14):
    text_path, video_path = config['trk_path'], config['mp4_path']

    cap = cv2.VideoCapture(video_path)

    df = pd.read_csv(text_path, sep=",", header=None)
    frame = 0
    
    cocodict = {}

    image_list = []
    object_cnt = max(df[12]) + 1


    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==frame,1:].values.tolist()

                
                for i in range(len(df_frame)):
                    if frame % 10 == 0:

                        frame_dict = {}
                        frame_dict['image_name'] = f'{frame//10}.jpg'
                        frame_dict['bbox'] = df_frame[i][:4]
                        frame_dict['class'] = 0
                        image_list.append(frame_dict)

                cv2.imwrite(f'images/{frame//10}.jpg',img)

                


                cv2.imshow(video_path, img)
                cv2.waitKey(2)


                frame+=1
            else:
                break

    else:
        print("can't open video")

    cap.release()
    cv2.destroyAllWindows()

    cocodict['annotations'] = image_list

    print(cocodict)

    with open ('test.json','w',encoding='utf-8') as make_file:
        json.dump(cocodict, make_file,indent='\t')




if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    frame_extraction(config)