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
info["date_created"] = "2021-11-18"

licenses["id"] = "1"
licenses["url"] = ""
licenses["name"] = "ALL RIGHTS RESERVED"

categories["id"] = "1"
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




def frame_extraction (config, PI = 3.14):
    text_path, video_path = config['trk_path'], config['mp4_path']

    cap = cv2.VideoCapture(video_path)

    df = pd.read_csv(text_path, sep=",", header=None)
    frame = 0

    
    object_cnt = max(df[12]) + 1

    img_list = []
    ann_list = []
    ann_id = 0
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
                image['id'] = frame
                image['filename'] = f'{frame}.jpg'
                # for i in range(len(df_frame)):
                    
                img_list.append(image)
                    
                for d in df_frame:
                    ann = {}
                    ann['id'] = ann_id
                    ann['iscrowd'] = 0
                    ann['bbox'] = d[:4]
                    ann['image_id'] = frame
                    ann['category_id'] = 1
                    ann_id += 1

                    ann_list.append(ann)
                
                cv2.imwrite(f'images/{frame}.jpg',img)

                


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
    # cocodict['annotations'] = image_list

    # print(cocodict)
    print(coco_group)
    with open ('test.json','w',encoding='utf-8') as make_file:
        json.dump(coco_group, make_file,indent='\t')




if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    frame_extraction(config)