import cv2
import pandas as pd
import numpy as np
import yaml
from Rotate_function import rotate_box_dot
from shapely.geometry import Polygon
import os

# 마우스로 다각형을 만드는 클래스
class draw_polygon :

    # 클래스 생성 시  
    def __init__(self,config):

        # 리스트 형으로 drink_section과 eat_section 초기화
        self.drink_section = []
        self.eat_section = []

        # config의 영상 첫 프레임을 가져와 drink_img, eat_img에 저장
        cap = cv2.VideoCapture(config['mp4_path'])
        self.drink_img =  cap.read()[1]
        self.eat_img = cap.read()[1]

        # drink와 eat의 각 이미지에서 다각형 점을 찍은 후 esc로 탈출, 찍은 점은 각 section 리스트에 저장
        self.select_section()


    # drink_img의 마우스 이벤트
    def drink_mouse_event(self, event, x, y, flags, param):

        # 왼쪽 클릭 시 클릭 좌표 저장 및 drink_img에 해당좌표 표시
        if event == cv2.EVENT_FLAG_LBUTTON :
            self.drink_section.append([x,y])
            cv2.circle(self.drink_img,(x,y),3,(255,0,0),2)
            cv2.imshow('drink',self.drink_img)
        
    # eat_img의 마우스 이벤트
    def eat_mouse_event(self, event, x, y, flags, param):

        # 왼쪽 클릭 시 클릭 좌표 저장 및 eat_img에 해당좌표 표시
        if event == cv2.EVENT_FLAG_LBUTTON :
            self.eat_section.append([x,y])
            cv2.circle(self.eat_img,(x,y),3,(0,0,255),2)
            cv2.imshow('eat',self.eat_img)

    # drink_img와 eat_img에 마우스 이벤트를 실행하기 위한 함수
    def select_section(self):
        
        cv2.imshow('drink',self.drink_img)
        cv2.setMouseCallback('drink',self.drink_mouse_event,self.drink_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


        cv2.imshow('eat',self.eat_img)
        cv2.setMouseCallback('eat',self.eat_mouse_event,self.eat_img)
        cv2.waitKey()
        cv2.destroyAllWindows()




def cnt_drink_eat(config):

    # drink section과 eat section 선택
    polygon_draw = draw_polygon(config)

    drink_section = polygon_draw.drink_section
    eat_section = polygon_draw.eat_section

    # 프레임 당 리터, 프레임 당 그램 변수 저장
    L_per_frame = config['L_per_frame']
    gram_per_frame = config['gram_per_frame']

    # 영상 가져오기
    text_path, video_path = config['trk_path'], config['mp4_path']

    cap = cv2.VideoCapture(video_path)
    out = video_out(video_path,os.path.splitext(os.path.basename(video_path))[0]+'trk.mp4')


    df = pd.read_csv(text_path, sep=",", header=None)
    
    object_cnt = max(df[12]) + 1
    object_dict = {i:[0,0] for i in range(object_cnt)}
    
    frame = 0
    if cap.isOpened():
        while True:
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==frame,1:].values.tolist()
                
                
                
                # drink section 다각형 그리기
                img = cv2.polylines(img,[np.array(drink_section)],True,(0,0,255),2)

                # eat section 다각형 그리기
                img = cv2.polylines(img,[np.array(eat_section)],True,(255,0,0),2)



                for i in df_frame:

                    
                    # 객체 rotate_box 그리기
                    box = rotate_box_dot(i[0], i[1], i[2], i[3], i[4])
                    img = cv2.polylines(img,[box],True,(255,255,255),thickness=3)


                    # 객체와 drink_section iou 계산
                    drink_iou, drink_bound = iou(drink_section, box)

                    # 객체와 eat_section iou 계산
                    eat_iou, eat_bound = iou(eat_section, box)

                    
                    
                    # config에서 지정한 iou보다 현재 프레임의 특정 개체와 drink_section간 iou가 큰 프레임을 카운트
                    if drink_iou > config['drink_iou'] :
                        # 식사량 frame count
                        object_dict[int(i[11])][0] += 1
                        # intersection 색칠
                        img = cv2.fillConvexPoly(img, np.int32([np.array(drink_bound)]), (100, 100, 255))
                        
                    # config에서 지정한 iou보다 현재 프레임의 특정 개체와 eat_section간 iou가 큰 프레임을 카운트
                    if eat_iou > config['eat_iou'] :
                        # 음수량 frame count
                        object_dict[int(i[11])][1] += 1
                        # intersection 색칠
                        img = cv2.fillConvexPoly(img, np.int32([np.array(eat_bound)]), (255, 100, 0))


                    img = cv2.putText(img, str(int(i[11])), (int(i[0]),int(i[1])), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)

                for i in object_dict.keys():
                    
                    img = cv2.putText(img, f'No.{i} drink: {round(object_dict[i][0]*L_per_frame,2)}L, eat: {round(object_dict[i][1]*gram_per_frame,2)}g', 
                    (1000, 40*(i+1)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)  

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
    
    print(object_dict)

    
    for i in object_dict.keys():
        print(f'{i}번 음수량 : {round(object_dict[i][0]*L_per_frame,2)}L  식사량 : {round(object_dict[i][1]*gram_per_frame,2)}g')



def iou (section,box):

    section_polygon = Polygon(section)
    box_polygon = Polygon([tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3])])

    intersection = section_polygon.intersection(box_polygon).area
    union = section_polygon.union(box_polygon).area
    bound = section_polygon.intersection(box_polygon).boundary

    return intersection/union, bound


def video_out(video_path,video_name):
    
    cap = cv2.VideoCapture(video_path) # VideoCapture 객체 정의
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # 코덱 정의

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(video_name, fourcc, fps, (int(width), int(height))) # VideoWriter 객체 정의

    
    return out



if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    cnt_drink_eat(config)
    