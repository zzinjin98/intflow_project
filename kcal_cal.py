import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot
import yaml
import random
import math
import os



def kcal_cal (config):
    text_path, video_path = config['trk_path'], config['mp4_path']

    cap = cv2.VideoCapture(video_path)
    df = pd.read_csv(text_path, sep=",", header=None)
    frame = 0
    object_cnt = max(df[12]) + 1


    kg_per_squaremeter = config['kg_per_squaremeter']
    move_length_to_kcal = config['move_length_to_kcal']
    m_per_pixel = config['m_per_pixel']

    # 돼지 수만큼의 딕셔너리 값 생성  1:소 번호 , 2:x좌표 , 3:y좌표 , 4:활동량 칼로리 누적 , 5:기초대사량 칼로리 누적 
    object_dict = {i:[0,0,0,0,0] for i in range(object_cnt)}


    # 최대 개체 수만큼 랜덤으로 컬러를 만듦
    color_list = [(int(random.random()*i*256) % 25,int(random.random()*i*256) % 256,int(random.random()*i*256) % 256) for i in range(1,object_cnt+1)]

    # 눈,목,꼬리 컬러
    color_nose = (255,255,255)
    color_neck = (255,0,0)
    color_tail = (255,255,0)


    # 각 객체별 높이, 너비 딕셔너리
    width_dict = {i:df.loc[df[12]==i, 3].tolist() for i in range(object_cnt)}
    height_dict = {i:df.loc[df[12]==i, 4].tolist() for i in range(object_cnt)}


    pd_width = pd.DataFrame.from_dict(width_dict, orient='index')
    pd_height = pd.DataFrame.from_dict(height_dict, orient='index')
    pd_width_res = pd_width.transpose()
    pd_height_res = pd_height.transpose()
    width_Q1 = []
    height_Q3 = []


    for i in range(object_cnt):
        Q1 = pd_width_res[i].quantile(q = 0.25, interpolation='nearest')
        Q3 = pd_height_res[i].quantile(q = 0.75, interpolation='nearest')
        width_Q1.append(float(Q1)*m_per_pixel)
        height_Q3.append(float(Q3)*m_per_pixel)

    

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


                    
                    if frame == 0:
                        object_dict[int(i[11])][0] = int(i[1])
                        object_dict[int(i[11])][1] = int(i[2])

                    
                    else:
                        move_length = math.dist(object_dict[int(i[11])][0:2], [int(i[1]),int(i[2])])
                        object_dict[int(i[11])][2] += move_length

                    object_dict[int(i[11])][3] += object_dict[int(i[11])][2] * move_length_to_kcal # 실시간 이동량 * 환산계수
                    
                    cow_surface = math.pi * float(width_Q1[int(i[11])]) * ((float(width_Q1[int(i[11])]) /2) + (float(height_Q3[int(i[11])])))
                    cow_bmr = float(cow_surface) * kg_per_squaremeter / (24*60*60*30)
                    object_dict[int(i[11])][4] += cow_bmr


                for i in range(object_cnt):
                    img = cv2.putText(img,f'No.{i} move_kcal: {round(object_dict[i][3],3)}',(1000, 40*(i+1)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)    
                    img = cv2.putText(img,f'No.{i} bmr_kcal: {round(object_dict[i][4],3)}',(1000, 40*(object_cnt+i+1)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)    

        
                cv2.imshow(video_path, img)
                cv2.waitKey(50)

                frame+=1
            else:
                break

    else:
        print("can't open video")

    cap.release()
    cv2.destroyAllWindows()


    for i in object_dict.keys():
        # etc_kcal == bmr_kcal / 4
        etc_kcal = object_dict[i][4] / 4
        print(f'{i}번 move 칼로리 : {round(object_dict[i][3],3)} , bmr 칼로리 : {round(object_dict[i][4],3)} , 총 칼로리 : {round(object_dict[i][3] + object_dict[i][4] + etc_kcal,3)}')



if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    kcal_cal(config)