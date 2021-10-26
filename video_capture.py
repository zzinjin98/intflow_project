import cv2
import numpy as np
import pandas as pd
from Rotate_function import rotate, rotate_box_dot

def read_text(text_url):
    # df변수에 데이터프레임으로 text파일을 가져온다
    df = pd.read_csv(text_url, sep=",", header=None)
    df.columns = ['frame','xc','yc','width','height','theta','no_x','no_y','ne_x','ne_y','ta_x','ta_y']
    df = df.drop(index=0, axis=0)


    rotate_list = []

    # 각 줄에 접근하여 roatate_function 좌표 계산에 필요한 데이터를 지정해준다
    
    for i in range(len(df)):
        data = list(df.loc[i+1,:])
        result = rotate_box_dot(int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]))
        rotate_list.append(result)

    #  출력 내용과 frame을 새로운 데이터 프레임에 저장한다
    df_rotate = pd.DataFrame(rotate_list, columns=['frame','rotated_x1','rotated_y1', 'rotated_x2','rotated_y2', 'rotated_x3','rotated_y3', 'rotated_x4','rotated_y4'])

    return df_rotate



def frame_extraction (text_url,video_url):

    df_rotate = read_text(text_url)

    cap = cv2.VideoCapture(video_url)
    frame = 0
    if cap.isOpened():
        while True:
            # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장된다
            ret, img = cap.read()
            
            if ret:
                
                # 현재 프레임을 가진 좌표 데이터를 가져온다
                df_0 = df_rotate[df_rotate['frame']==frame]
                df_array_0 = []

                # 가져온 데이터의 인덱스를 0부터로 초기화한다
                df_0.index = [i for i in range(len(df_0))]

                # 좌표 데이터를 가져와 4행 2열 형태로 바꾼 뒤 df_array_0에 넣는다
                for i in range(len(df_0)):
                    df_array_0.append(np.reshape(list(df_0.loc[i,:])[1:], (4,2)))

                # df_array_0 리스트 안 points의 각 네 점으로 현재 프레임이미지에 다각형을 그린다
                color = 255
                for points in df_array_0 :
                    
                    img = cv2.polylines(img,[points],True,(color,color,color),thickness=3)


                cv2.imshow(video_url, img)
                cv2.waitKey(25)

                frame+=1
            else:
                break

    else:
        print("can't open video")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_extraction("..\mounting_000\mounting_000_det.txt", "..\mounting_000\mounting_000.mp4")
