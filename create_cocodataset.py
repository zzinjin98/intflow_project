import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot
import yaml
import random
import math
import os
import json
from collections import OrderedDict

coco_group = OrderedDict()

info = OrderedDict()
licenses = OrderedDict()
categories = OrderedDict()
images = OrderedDict()
annotations = OrderedDict()



info['year'] = '2021'
info["version"] = "0.1.0"
info["description"] = "Dataset in COCO Format"
info["contributor"] = ""
info["url"] = ""
info["date_created"] = "2021-11"

licenses["id"] = 1
licenses["url"] = ""
licenses["name"] = "ALL RIGHTS RESERVED"

categories["id"] = 1
categories["name"] = "cow"
categories["supercategory"] = "none"

annotations = "{}"

im_name_list = []
a_list = []

coco_group["info"] = info
coco_group["licenses"] = [licenses]
coco_group["categories"] = [categories]
coco_group["images"] = [images]
coco_group["annotations"] = [annotations]

categories

print(json.dumps(coco_group, ensure_ascii=False, indent="\t"))


def circumscription (points):
    x_min = min(points[:,0])
    x_max = max(points[:,0])
    y_min = min(points[:,1])
    y_max = max(points[:,1])

    print(points[:,0])
    print(points[:,1])

    w = x_max - x_min
    h = y_max - y_min
    return float(w),float(h)

def frame_extraction (config, PI = 3.14):
    text_path, video_path = config['trk_path'], config['mp4_path']

    cap = cv2.VideoCapture(video_path)

    df = pd.read_csv(text_path, sep=",", header=None)
    frame = 0

    img_cnt = 1
    object_cnt = max(df[12]) + 1

    img_list = []
    ann_list = []
    ann_id = 1
    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                df_frame = df.loc[df[0]==frame,1:].values.tolist()

                h,w,c = img.shape
                image = {}
                image['height'] = h
                image['width'] = w
                image['license'] = 1
                image['id'] = img_cnt
                image['filename'] = f'{img_cnt}.jpg'
                    
                img_list.append(image)
                
                if frame % 50 == 0 :
                    for d in df_frame:
                        points = rotate_box_dot(d[0], d[1], d[2], d[3], d[4])
                        bbox_w,bbox_h =circumscription(points)

                        ann = {}
                        ann['id'] = ann_id
                        ann['iscrowd'] = 0
                        ann['bbox'] = d[:2]+[w,h]
                        ann['image_id'] = img_cnt
                        ann['category_id'] =1
                        ann['area'] = bbox_w*bbox_h

                        ann_id += 1

                        ann_list.append(ann)
                        cv2.rectangle(img, (int(d[0]-bbox_w/2),int(d[1]-bbox_h/2)),(int(d[0]+bbox_w/2),int(d[1]+bbox_h/2)),thickness=3,color=(255,255,255))
                        img = cv2.polylines(img,[points],True,(255,0,0),thickness=2)

                    cv2.imshow(video_path, img)

                    cv2.waitKey(1000)
                        
                

                    cv2.imwrite(f'images/{img_cnt}.jpg',img)

                    img_cnt += 1

                    
                cv2.imshow(video_path, img)
                cv2.waitKey(2)


                frame+=1
            else:
                break

    else:
        print("can't open video")

    cap.release()
    cv2.destroyAllWindows()

    coco_group['images'] = img_list
    coco_group['annotations'] = ann_list

    
    with open ('test.json','w',encoding='utf-8') as make_file:
        json.dump(coco_group, make_file,indent='\t')




if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    frame_extraction(config)