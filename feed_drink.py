import cv2
import pandas as pd
import yaml
from Rotate_function import rotate_box_dot
from shapely.geometry import Polygon

def select_section(config):
    video_path = config['mp4_path']

    cap = cv2.VideoCapture(video_path)

    ret,img = cap.read()

    x,y,w,h = cv2.selectROI('image',img,False)

    if w and h :
        roi = img[y:y+h, x:x+w]
        cv2.imshow('drag',roi)
    
    cv2.imshow('image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return [x,y,w,h]

def cnt_drink_eat(config):

    drink_section = select_section(config)
    eat_section = select_section(config)
    
    L_per_frame = config['L_per_frame']
    gram_per_frame = config['gram_per_frame']

    text_path, video_path = config['trk_path'], config['mp4_path']

    cap = cv2.VideoCapture(video_path)
    df = pd.read_csv(text_path, sep=",", header=None)
    
    object_cnt = max(df[12]) + 1
    object_dict = {i:[0,0] for i in range(object_cnt)}
    
    frame = 0
    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==frame,1:].values.tolist()
                
                
                
                # drink section rectangle
                img = cv2.rectangle(img,(drink_section[0],drink_section[1]),(drink_section[0]+drink_section[2],drink_section[1]+drink_section[3]),(255,0,0),2)

                # eat section rectangle
                img = cv2.rectangle(img,(eat_section[0],eat_section[1]),(eat_section[0]+eat_section[2],eat_section[1]+eat_section[3]),(0,255,0),2)

                

                for i in df_frame:

                    
                    # 직사각형
                    # box = [i[0]-i[2]/2,i[1]-i[3]/2,i[0]+i[2]/2,i[1]+i[3]/2]
                    # img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,255,255),2)


                    # 다각형
                    box = rotate_box_dot(i[0], i[1], i[2], i[3], i[4])
                    img = cv2.polylines(img,[box],True,(255,255,255),thickness=3)

                    drink_iou = iou(drink_section,box)
                    eat_iou = iou(eat_section,box)

                    if drink_iou > config['drink_iou'] :
                        object_dict[int(i[11])][0] += 1
                    
                    if eat_iou > config['eat_iou'] :
                        object_dict[int(i[11])][1] += 1


                    img = cv2.putText(img, str(int(i[11])), (int(i[0]),int(i[1])), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)  

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
        print(f'{i}번 음수량 : {round(object_dict[i][0]*L_per_frame,2)}L  식사량 : {round(object_dict[i][1]*gram_per_frame,2)}g')



def iou (section,box):

    section_polygon = Polygon([(section[0],section[1]),(section[0]+section[2],section[1]),(section[0]+section[2],section[1]+section[3]),(section[0],section[1]+section[3])])
    box_polygon = Polygon([tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3])])


    return section_polygon.intersection(box_polygon).area / section_polygon.union(box_polygon).area



if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    cnt_drink_eat(config)