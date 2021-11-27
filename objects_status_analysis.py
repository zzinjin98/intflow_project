import cv2
import pandas as pd
import numpy as np
import yaml
from Rotate_function import rotate_box_dot
from shapely.geometry import Polygon, Point
import os, random, math

# 마우스로 다각형을 만드는 클래스
class draw_polygon :
    
    # 클래스 생성 시  
    def __init__(self,config):
        # 리스트 형으로 drink_section과 eat_section 초기화'
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

# 객체의 활동량, 급이량, 음수량 산출
def objects_status_analysis(config):
    
    # drink section과 eat section 선택
    polygon_draw = draw_polygon(config)
    
    # section 좌표 저장
    drink_section = polygon_draw.drink_section
    eat_section = polygon_draw.eat_section
    
    # config 변수 가져오기
    kg_per_squaremeter = config['kg_per_squaremeter']
    move_length_to_kcal = config['move_length_to_kcal']
    m_per_pixel = config['m_per_pixel']
    ml_per_frame = config['ml_per_frame']
    gram_per_frame = config['gram_per_frame']
    
    # 영상 가져오기
    text_path, video_path = config['trk_path'], config['mp4_path']
    cap = cv2.VideoCapture(video_path)
    
    # 영상 저장경로 및 이름 설정
    out = video_out(video_path,os.path.splitext(os.path.basename(video_path))[0]+'trk.mp4')
    
    # txt파일을 데이터 프레임으로 저장
    df = pd.read_csv(text_path, sep=",", header=None)
    
    # 영상내 객체 최대 마리수 저장
    object_cnt = max(df[12]) + 1
    object_dict = {i:[0,0] for i in range(object_cnt)}
    
    # 돼지 수만큼의 딕셔너리 값 생성  1:소 번호 , 2:x좌표 , 3:y좌표 , 4:활동량 칼로리 누적 , 5:기초대사량 칼로리 누적 , 6:음수량 , 7: 식사량
    object_dict = {i:[0,0,0,0,0,0,0] for i in range(object_cnt)}
    
    # 최대 개체 수만큼 랜덤으로 컬러를 만듦
    color_list = [(int(random.random()*i*256) % 25,int(random.random()*i*256) % 256,int(random.random()*i*256) % 256) for i in range(1,object_cnt+1)]
    
    # 눈,목,꼬리 컬러
    color_nose = (255,255,255)
    color_neck = (255,0,0)
    color_tail = (255,255,0)
    
    # 각 객체별 높이, 너비 딕셔너리
    width_dict = {i:df.loc[df[12]==i, 3].tolist() for i in range(object_cnt)}
    height_dict = {i:df.loc[df[12]==i, 4].tolist() for i in range(object_cnt)}
    
    # 높이, 너비 딕셔너리를 DataFame 형태로 변형
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
                    
                    # 몸통 다각형
                    points = rotate_box_dot(i[0], i[1], i[2], i[3], i[4])
                   
                    # img = cv2.polylines(img,[points],True,color_list[int(i[11])],thickness=3)
                    img = cv2.polylines(img,[points],True,(255,255,255),thickness=3)
                    
                    # 몸통 타원
                    #img = cv2.ellipse(img,((i[0],i[1]),(i[2],i[3]),float(i[4])*360/math.pi),color_list[int(i[11])],thickness=3)
                    
                    # 코
                    img = cv2.circle(img,(int(i[5]),int(i[6])), 5,color_nose, thickness=2)
                    
                    # 목
                    img = cv2.circle(img,(int(i[7]),int(i[8])), 5,color_neck, thickness=2)
                   
                    # 꼬리
                    img = cv2.circle(img,(int(i[9]),int(i[10])), 5,color_tail, thickness=2)
                   
                    # 코 목 선
                    cv2.line(img, (int(i[5]),int(i[6])), (int(i[7]), int(i[8])),color_list[int(i[11])],thickness=2)
                    length1 = round(math.dist([int(i[5]),int(i[6])],[int(i[7]),int(i[8])]),3)
                   
                    # img = cv2.putText(img, str(length1), (int((int(i[5])+int(i[7]))/2),int((int(i[6])+int(i[8]))/2)), cv2.FONT_HERSHEY_PLAIN, 1, color_list[int(i[11])], thickness=2)    
                    # 목 꼬리 선
                    cv2.line(img, (int(i[7]), int(i[8])), (int(i[9]),int(i[10])),color_list[int(i[11])],thickness=2)
                    length2 = round(math.dist([int(i[7]),int(i[8])],[int(i[9]),int(i[10])]),3)
                   
                    # img = cv2.putText(img, str(length2), (int((int(i[7])+int(i[9]))/2),int((int(i[8])+int(i[10]))/2)), cv2.FONT_HERSHEY_PLAIN, 1, color_list[int(i[11])], thickness=2)
                    # 객체 번호
                    img = cv2.putText(img, str(int(i[11])), (int(i[0]),int(i[1])), cv2.FONT_HERSHEY_PLAIN, 2, color_list[int(i[11])], thickness=2)    
                   
                    # 객체의 코, 목 좌표를 지름으로 하는 원 생성
                    img = cv2.circle(img, (int((i[5]+i[7])/2), int((i[6]+i[8])/2)),
                                     int(round(math.dist([int(i[5]),int(i[6])],[int(i[7]),int(i[8])]),3)/2), (255,255,255), 2)
                    circle = [(int((i[5]+i[7])/2), int((i[6]+i[8])/2)),
                              int(round(math.dist([int(i[5]),int(i[6])],[int(i[7]),int(i[8])]),3)/2)]
                    
                    # 객체와 drink_section iou 계산
                    # drink_iou, drink_bound = iou_box(drink_section, box)
                    drink_iou, drink_bound = iou_circle(drink_section, circle)
                    # 객체와 eat_section iou 계산
                    # eat_iou, eat_bound = iou_box(eat_section, box)
                    eat_iou, eat_bound = iou_circle(eat_section, circle)                  
                    # config에서 지정한 iou보다 현재 프레임의 특정 개체와 drink_section간 iou가 큰 프레임을 카운트
                   
                    if drink_iou > config['drink_iou'] :
                        # 식사량 frame count
                        object_dict[int(i[11])][5] += 1
                        # intersection 색칠
                        img = cv2.fillConvexPoly(img, np.int32([np.array(drink_bound)]), (100, 100, 255))
                    # config에서 지정한 iou보다 현재 프레임의 특정 개체와 eat_section간 iou가 큰 프레임을 카운트
                    
                    if eat_iou > config['eat_iou'] :
                        # 음수량 frame count
                        object_dict[int(i[11])][6] += 1
                        # intersection 색칠
                        img = cv2.fillConvexPoly(img, np.int32([np.array(eat_bound)]), (255, 100, 0))
                    if frame == 0:
                        object_dict[int(i[11])][0] = int(i[1])
                        object_dict[int(i[11])][1] = int(i[2])                
                    else:
                        move_length = math.dist(object_dict[int(i[11])][0:2], [int(i[1]),int(i[2])])
                        object_dict[int(i[11])][2] += move_length
                    object_dict[int(i[11])][3] += object_dict[int(i[11])][2] * move_length_to_kcal # 실시간 이동거리 * 환산계수
                    cow_surface = math.pi * float(width_Q1[int(i[11])]) * ((float(width_Q1[int(i[11])]) /2) + (float(height_Q3[int(i[11])])))
                    cow_bmr = float(cow_surface) * kg_per_squaremeter / (24*60*60*30)
                    object_dict[int(i[11])][4] += cow_bmr
                    img = cv2.putText(img, str(int(i[11])), (int(i[0]),int(i[1])), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)
                for i in object_dict.keys():
                    img = cv2.putText(img, f'No.{i} drink: {round(object_dict[i][5]*ml_per_frame,2)}ml, eat: {round(object_dict[i][6]*gram_per_frame,2)}g, move:{round(object_dict[i][3],2)}kcal, bmr:{round(object_dict[i][4],2)}kcal',
                    (0, 30*(i+1)), cv2.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), thickness=2)
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
    
    for i in object_dict.keys():
        etc_kcal = object_dict[i][4] / 4
        print(f'{i}번 음수량 : {round(object_dict[i][5]*ml_per_frame,2)}L, 식사량 : {round(object_dict[i][6]*gram_per_frame,2)}g, move: {round(object_dict[i][3],2)}kcal , bmr: {round(object_dict[i][4],3)}kcal, 총 칼로리 : {round(object_dict[i][3] + object_dict[i][4] + etc_kcal,2)}kcal')
        
# object(box)와 section이 겹치는 비율(iou) 산출
def iou_box (section, box):
    section_polygon = Polygon(section)
    box_polygon = Polygon([tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3])])
    intersection = section_polygon.intersection(box_polygon).area
    union = section_polygon.union(box_polygon).area
    bound = section_polygon.intersection(box_polygon).boundary
    return intersection/union, bound

# object(circle)와 section이 겹치는 비율(iou) 산출
def iou_circle (section, circle):
    section_polygon = Polygon(section)
    circle_polygon = Point(circle[0]).buffer(circle[1])
    intersection = section_polygon.intersection(circle_polygon).area
    union = section_polygon.union(circle_polygon).area
    bound = section_polygon.intersection(circle_polygon).boundary
    return intersection/union, bound

# 결과 video 저장
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
    objects_status_analysis(config)