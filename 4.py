import dlib
import os
import cv2
import math

from criteria import shape_loss


predictor_path  = "D:/SAM-master/shape_predictor_68_face_landmarks.dat"
png_path = "D:/SAM-master/CelebAMask-HQ/test_img/1.jpg"


detector = dlib.get_frontal_face_detector()
#相撞
predicator = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()
img1 = cv2.imread(png_path)

dets = detector(img1,1)
print("Number of faces detected : {}".format(len(dets)))
for k,d in enumerate(dets):
    print("Detection {}  left:{}  Top: {} Right {}  Bottom {}".format(
        k,d.left(),d.top(),d.right(),d.bottom()
    ))
    lanmarks = [[p.x,p.y] for p in predicator(img1,d).parts()]
    for idx,point in enumerate(lanmarks):
        point = (point[0],point[1])
        cv2.circle(img1,point,5,color=(0,0,255))
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(img1,str(idx),point,font,0.5,(0,255,0),1,cv2.LINE_AA)
        #对标记点进行递归；


    point7 = lanmarks[6]
    point8 = lanmarks[7]
    point9 = lanmarks[8]

    distance = math.sqrt((point8[0] - point7[0])**2 + (point8[1] - point7[1])**2)
    distance2 = math.sqrt((point9[0] - point8[0])**2 + (point9[1] - point8[1])**2)

    print("Distance between point 7 and point 8:", distance)
    print("Distance between point 8 and point 9:", distance2)


cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img",img1)
cv2.waitKey(0)
