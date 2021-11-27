import cv2
import pandas as pd
from Rotate_function import  rotate_box_dot
import yaml
import os
import json
from collections import OrderedDict


# rotate box의 네 좌표를 입력받아 box에 외접하는 직사각형의 너비와 높이 추출
def circumscription (points):
    x_min = min(points[:,0])
    x_max = max(points[:,0])
    y_min = min(points[:,1])
    y_max = max(points[:,1])

    w = x_max - x_min
    h = y_max - y_min

    return float(w),float(h)


# 소 영상과 trk.txt를 입력받아 50 frame마다 img를 저장하고 해당 이미지에 대한 정보를 누적하여
# annotation.json 파일을 생성하는 함수
def create_img_ann (config,switch):
		

	# 가장 바깥의 딕셔너리
    coco_group = OrderedDict()
		
	# 내부 5가지 정보 딕셔너리
    info = OrderedDict()
    licenses = OrderedDict()
    categories = OrderedDict()
    images = OrderedDict()
    annotations = OrderedDict()


	# info, licenses 는 학습에 사용되지 않으므로 임의로 설정
    info['year'] = '2021'
    info["version"] = "0.1.0"
    info["description"] = "Dataset in COCO Format"
    info["contributor"] = ""
    info["url"] = ""
    info["date_created"] = "2021-11"

    licenses["id"] = 1
    licenses["url"] = ""
    licenses["name"] = "ALL RIGHTS RESERVED"

	# categories 는 모든 이미지에서 소만 탐지하면 되기 때문에 다음과 같이 설정
    categories["id"] = 1
    categories["name"] = "cow"
    categories["supercategory"] = "none"

	# info, licenses, categories를 가장 바깥 딕셔너리에 저장
    coco_group["info"] = info
    coco_group["licenses"] = [licenses]
    coco_group["categories"] = [categories]

	# switch가 train이면 train 데이터, test면 test 데이터 경로를 가져옴
    if switch == 'train':
        data_path = config['train_path']
    else:
        data_path = config['test_path']
    
    data_list = os.listdir(data_path)

		
    img_cnt = 1
    img_list = []
    ann_list = []
    ann_id = 1

	# train 혹은 test 경로에 있는 모든 폴더에 접근하여 소 영상과 trk.txt를 통해 데이터를 생성
    for t in data_list:
        
        # video_path ex : train_data/1/1.mp4 , text_path ex : train_data/1/1_trk.txt
        video_path, text_path  = map(lambda x : data_path+'/'+t+'/'+x ,os.listdir(data_path+'/'+t))

        cap = cv2.VideoCapture(video_path)

        df = pd.read_csv(text_path, sep=",", header=None)
        frame = 0
        
        if cap.isOpened():
            while True:
                # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
                ret, img = cap.read()
                
                if ret:
                    # frame에 맞는 객체들의 정보를 모두 가져옴
                    df_frame = df.loc[df[0]==frame,1:].values.tolist()
											
                    # 50 frame 단위로 다음과 같은 작업 실행
                    if frame % 50 == 0 :

                        # img정보를 images에 저장
                        h,w,c = img.shape
                        image = {}
                        image['height'] = h
                        image['width'] = w
                        image['license'] = 1
                        image['id'] = img_cnt
                        image['file_name'] = f'{img_cnt}.jpg'
                            
                        img_list.append(image)

                        # 현재 frame에 존재하는 각 소들의 정보를 annotation에 저장
                        for d in df_frame:

                            # trk.txt에 있는 w,h는 rotate_box 크기이고 필
                            points = rotate_box_dot(d[0], d[1], d[2], d[3], d[4])
                            bbox_w,bbox_h =circumscription(points)

                            ann = {}
                            ann['id'] = ann_id
                            ann['iscrowd'] = 0
                            ann['bbox'] = d[:2]+[bbox_w,bbox_h]  # bbox는 객체를 포함하는 직사각형 [xc, yc, w, h]
                            ann['image_id'] = img_cnt
                            ann['category_id'] =1
                            ann['area'] = float(d[2]*d[3]) # area는 segmentation의 넓이

														# rotate box를 segmentation으로 활용하여 정확도를 높임
                            ann['segmentation'] =[[max(0.0,i) for i in map(float,points.flatten())]]

                            ann_id += 1

                            ann_list.append(ann)
                        #     cv2.rectangle(img, (int(d[0]-bbox_w/2),int(d[1]-bbox_h/2)),(int(d[0]+bbox_w/2),int(d[1]+bbox_h/2)),thickness=3,color=(255,255,255))
                        # cv2.imshow(video_path, img)

                        # cv2.waitKey(1000)
                        
                        # trian_img 혹은 test_img에 현재 이미지 저장
                        if switch == 'train':

                            if not os.path.exists('train_img'):
                                os.makedirs('train_img')

                            cv2.imwrite(f'train_img/{img_cnt}.jpg',img)
                        
                        else:
                            if not os.path.exists('test_img'):
                                os.makedirs('test_img')

                            cv2.imwrite(f'test_img/{img_cnt}.jpg',img)

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
		
    # 누적한 img 와 annotation 정보를 저장
    coco_group['images'] = img_list
    coco_group['annotations'] = ann_list

    # train_annotation.json 혹은 test_annotation.json으로 저장
    if switch =='train':
        with open ('train_annotation.json','w',encoding='utf-8') as make_file:
            json.dump(coco_group, make_file,indent='\t')
    
    else:
        with open ('test_annotation.json','w',encoding='utf-8') as make_file:
            json.dump(coco_group, make_file,indent='\t')




if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    create_img_ann(config,'train')
    create_img_ann(config,'test')