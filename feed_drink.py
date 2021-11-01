import cv2
import pandas as pd
import yaml
import random


def iou (section,box):
    section_roi = (section[2]-section[0]+1) * (section[3]-section[1]+1)
    box_roi = (box[2]-box[0]+1) * (box[3]-box[1]+1)

    x1 = max(section[0],box[0])
    y1 = max(section[1],box[1])
    x2 = min(section[2],box[2])
    y2 = min(section[3],box[3])

    w = max(0,x2-x1+1)
    h = max(0,y2-y1+1)

    inter = w*h

    return inter / (section_roi+box_roi-inter)


def cnt_drink_eat(config):

    drink_section = config['drink_section']
    eat_section = config['eat_section']

    text_path, video_path = config['trk_path'], config['mp4_path']

    cap = cv2.VideoCapture(video_path)
    df = pd.read_csv(text_path, sep=",", header=None)
    
    object_cnt = max(df[12]) + 1
    object_dict = {i:[0,0] for i in range(object_cnt)}
    color_list = [(0, random.randrange(0,256), random.randrange(0,256)) for _ in range(object_cnt)]
    
    frame = 0
    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==frame,1:].values.tolist()
                
                # drink section rectangle
                img = cv2.rectangle(img,(drink_section[0],drink_section[1]),(drink_section[2],drink_section[3]),(255,0,0),2)

                # eat section rectangle
                img = cv2.rectangle(img,(eat_section[0],eat_section[1]),(eat_section[2],eat_section[3]),(0,255,0),2)

                for i in df_frame:
                    box = [i[0]-i[2]/2,i[1]-i[3]/2,i[0]+i[2]/2,i[1]+i[3]/2]
                    
                    # 객체 박스
                    img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,255,255),2)

                    # 객체 번호
                    img = cv2.putText(img, str(int(i[11])), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_PLAIN, 2, color_list[int(i[11])], thickness=2)    
                
                    drink_iou = iou(drink_section,box)
                    eat_iou = iou(eat_section,box)

                    if drink_iou > 0.05 :
                        object_dict[int(i[11])][0] += 1
                    
                    if eat_iou > 0.05 :
                        object_dict[int(i[11])][1] += 1


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
        print(f'{i}번 음수량 : {round(object_dict[i][0]*0.01,2)}L  식사량 : {round(object_dict[i][1]*0.1,2)}g')


if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    cnt_drink_eat(config)