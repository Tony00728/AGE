import dlib
import os
import cv2
import math
from options.train_options import TrainOptions
from PIL import Image
import numpy as np

opts = TrainOptions().parse()  # TrainOptions調用 parse 方法來解析命令行參數，並將結果存儲在 opts 變數中


class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def detect_landmarks(self, pil_img ):
        #img = np.array(pil_img)

        img = cv2.imread(pil_img)  #這
        dets = self.detector(img, 1)
        #print("Number of faces detected : {}".format(len(dets)))
        point4 = 0
        point5 = 0
        point6 = 0
        point7 = 0
        point8 = 0
        point9 = 0
        point10 = 0
        point11 = 0
        point12 = 0

        for k, d in enumerate(dets):
            #print("Detection {}  left:{}  Top: {} Right {}  Bottom {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()
            #))
            landmarks = [[p.x, p.y] for p in self.predictor(img, d).parts()]
            #for idx, point in enumerate(landmarks):
            #    point = (point[0], point[1])
            #    cv2.circle(img, point, 5, color=(0, 0, 255))
            #    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            #    cv2.putText(img, str(idx), point, font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            point4 = landmarks[4]
            point5 = landmarks[5]
            point6 = landmarks[6]
            point7 = landmarks[7]
            point8 = landmarks[8]
            point9 = landmarks[9]
            point10 = landmarks[10]
            point11 = landmarks[11]
            point12 = landmarks[12]

            #distance = math.sqrt((point8[0] - point7[0]) ** 2 + (point8[1] - point7[1]) ** 2)
            #distance2 = math.sqrt((point9[0] - point8[0]) ** 2 + (point9[1] - point8[1]) ** 2)

            #print("Distance between point 7 and point 8:", distance)
            #print("Distance between point 8 and point 9:", distance2)

        #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        #cv2.imshow("img", img)
        #cv2.waitKey(0)

        return [point4, point5, point6, point7, point8, point9, point10, point11, point12]



if __name__ == "__main__":

    #predictor_path = "/home/tony/SAM-master/shape_predictor_68_face_landmarks.dat"
    #png_path = "/home/tony/morph_train/image_new/019066_1M68.jpg"
    image_folder = "/home/tony/morph_train/40-49_resize256x256"

    landmark_detector = FaceLandmarkDetector(opts.face_landmarks_path)
    #landmark_detector.detect_landmarks(png_path)
    #x = landmark_detector.detect_landmarks(png_path)
    #print(x)

    point4 = []
    point5 = []
    point6 = []
    point7 = []
    point8 = []
    point9 = []
    point10 = []
    point11 = []
    point12 = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
            img_path = os.path.join(image_folder, filename)
            landmarks = landmark_detector.detect_landmarks(img_path)
            if landmarks:  # 檢查是否有偵測到人臉

                point4.append(landmarks[0])
                point5.append(landmarks[1])
                point6.append(landmarks[2])
                point7.append(landmarks[3])
                point8.append(landmarks[4])
                point9.append(landmarks[5])
                point10.append(landmarks[6])
                point11.append(landmarks[7])
                point12.append(landmarks[8])
            else:
                print(f"No face detected in image: {img_path}")
                os.remove(img_path)

    x_coordinates = [point[0] for point in point7]    #
    y_coordinates = [point[1] for point in point7]    #

    mean_x = sum(x_coordinates) / len(x_coordinates)
    mean_y = sum(y_coordinates) / len(y_coordinates)

    print("Mean X Coordinate:", mean_x)
    print("Mean Y Coordinate:", mean_y)



