import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot
import yaml
import random
import math
import os




def cow_bmr (config):
    text_path, video_path = config['trk_path'], config['mp4_path']

   
    cap = cv2.VideoCapture(video_path)

    df = pd.read_csv(text_path, sep=",", header=None)
    frame = 0

    object_cnt = max(df[12]) + 1

    object_dict = {i:[0,0,0] for i in range(object_cnt)}


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
                    #img = cv2.ellipse(img,((i[0],i[1]),(i[2],i[3]),float(i[4])*360/PI),color_list[int(i[11])],thickness=3)

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

                    # kcal 환산
                    # 다음 프레임의 좌표와의 거리 구하기
                    # 왜 거리좌표가 같음(움직임 0)에도 값이 저렇게 크게 나오는가??
                    # move_kcal 에는 각돼지의 움직임이 합산되어 버림-> 돼지마다의 움직임을 각자 계산되어야한다.
                    
                    if frame == 0:
                        object_dict[int(i[11])][0] = int(i[9])
                        object_dict[int(i[11])][1] = int(i[10])

                    
                    else:
                        move_length = math.dist(object_dict[int(i[11])][0:2], [int(i[9]),int(i[10])])
                        object_dict[int(i[11])][2] += move_length
                
                    

                    #img = cv2.putText(img, str(move_kcal), (int((int(i[9])+ int(i[10]))/2)), cv2.FONT_HERSHEY_PLAIN, 2, (144,144,144),thickness=2)


                    
                   
                cv2.imshow(video_path, img)

                
                cv2.waitKey(50)

                frame+=1
            else:
                break

    else:
        print("can't open video")

    cap.release()
    cv2.destroyAllWindows()

    print(object_dict)

    for i in object_dict.keys():
        print(f'{i}번 이동 량 : {round(object_dict[i][2])}')


if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    cow_bmr(config)
