import cv2

video_file = "intflow_project\mounting_001\mounting_001.mp4"

cap = cv2.VideoCapture(video_file)
frame = 0
if cap.isOpened():
    while True:
        # 프레임을 잘 읽어오면 ret는 True, img 는 프레임 이미지로 저장됨
        ret, img = cap.read()
        if ret:
            
            # 현재 프레임에 대한 여러 동물 객체 정보를 가져온다
            # 각 객체 정보에 대한 좌표를 계산한다
            # 계산된 rotate 좌표를 가지고 cv2.polylines 함수를 사용하여 다각형을 그린다
            # 그린 영상을 출력한다
            # 저장한다

            cv2.imshow(video_file, img)
            cv2.waitKey(25)

            frame+=1
        else:
            break


else:
    print("can't open video")

cap.release()
cv2.destroyAllWindows()