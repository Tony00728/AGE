import dlib
import os
import cv2
import math
from options.train_options import TrainOptions
from PIL import Image
import numpy as np

opts = TrainOptions().parse()  # TrainOptions調用 parse 方法來解析命令行參數，並將結果存儲在 opts 變數中

#白人女性
'''白人女性
class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def detect_landmarks(self, pil_img, input_age_range=None, target_age_range=None):
        img = np.array(pil_img)
        dets = self.detector(img, 1)


        # 判断是否跳过检测
        if input_age_range < 0.1 and target_age_range < 0.1:
            return []
        if 0.1 <= input_age_range < 0.2 and 0.1 <= target_age_range < 0.2:
            return []
        if 0.2 <= input_age_range < 0.3 and 0.2 <= target_age_range < 0.3:
            return []
        if 0.3 <= input_age_range < 0.4 and 0.3 <= target_age_range < 0.4:
            return []
        if 0.4 <= input_age_range < 0.5 and 0.4 <= target_age_range < 0.5:
            return []
        if 0.5 <= input_age_range < 0.6 and 0.5 <= target_age_range < 0.6:
            return []
        if input_age_range >= 0.6 and target_age_range >= 0.6:
            return []

        if target_age_range < 0.1:
            target_age_range = (0.0, 0.1)
        elif 0.1 <= target_age_range < 0.2:
            target_age_range = (0.1, 0.2)
        elif 0.2 <= target_age_range < 0.3:
            target_age_range = (0.2, 0.3)
        elif 0.3 <= target_age_range < 0.4:
            target_age_range = (0.3, 0.4)
        elif 0.4 <= target_age_range < 0.5:
            target_age_range = (0.4, 0.5)
        elif 0.5 <= target_age_range < 0.6:
            target_age_range = (0.5, 0.6)
        else:
            target_age_range = (0.6, 0.8)

        if input_age_range < 0.1:
            #輸入年齡10-
            age_specific_points = {
                (0.1, 0.2): [8, 9, 7, 10, 6, 11, 30, 5, 57, 58, 56, 33, 59, 66, 67, 32, 55, 34, 65, 61],
                (0.2, 0.3): [8, 9, 7, 10, 6, 11, 57, 5, 58, 56, 30, 55, 59, 33, 12, 66, 67, 65, 32, 61],
                (0.3, 0.4): [8, 9, 7, 10, 6, 11, 5, 12, 57, 58, 56, 66, 67, 65, 4, 55, 59, 13, 30, 33],
                (0.4, 0.5): [8, 9, 7, 10, 6, 11, 5, 12, 4, 13, 3, 14, 56, 57, 65, 55, 66, 58, 67, 2],
                (0.5, 0.6): [ 9, 8, 7, 10, 6, 11, 5, 12, 13, 4, 14, 3, 15, 0, 16, 61, 2, 62, 67, 59],
                (0.6, 0.8): [9, 10, 8, 7, 6, 11, 5, 12, 4, 13, 3, 14, 2, 16, 15, 0, 1, 65, 66, 67]
            }
        elif 0.1 <= input_age_range < 0.2:
            #輸入年齡20-29
            age_specific_points = {
                (0.0, 0.1): [8, 9, 7, 10, 6, 11, 30, 5, 57, 58, 56, 33, 59, 66, 67, 32, 55, 34, 65, 61],
                (0.2, 0.3): [10, 9, 8, 7, 11, 6, 12, 0, 17, 57, 5, 56, 13, 58, 14, 55, 59, 4, 18, 48],
                (0.3, 0.4): [10, 9, 8, 7, 6, 11, 5, 12, 4, 13, 14, 3, 65, 66, 56, 15, 57, 67, 55, 58],
                (0.4, 0.5): [10, 11, 5, 6, 9, 8, 7, 4, 12, 13, 3, 14, 2, 15, 16, 1, 0, 65, 54, 55],
                (0.5, 0.6): [11, 10, 12, 13, 6, 5, 14, 9, 7, 15, 8, 16, 4, 3, 0, 1, 2, 48, 54, 17],
                (0.6, 0.8): [11, 5, 10, 6, 4, 12, 13, 3, 9, 16, 14, 15, 8, 7, 2, 1, 0, 54, 26, 17]
            }

        elif 0.2 <= input_age_range < 0.3:
            #輸入年齡20-29
            age_specific_points = {
                (0.0, 0.1): [8, 9, 7, 10, 6, 11, 57, 5, 58, 56, 30, 55, 59, 33, 12, 66, 67, 65, 32, 61],
                (0.1, 0.2): [10, 9, 8, 7, 11, 6, 12, 0, 17, 57, 5, 56, 13, 58, 14, 55, 59, 4, 18, 48],
                (0.3, 0.4): [5, 6, 4, 10, 8, 7, 9, 11, 3, 2, 12, 1, 0, 13, 48, 60, 14, 65, 64, 15],
                (0.4, 0.5): [5, 4, 3, 2, 6, 11, 1, 0, 10, 12, 13, 16, 9, 15, 7, 14, 8, 17, 26, 30],
                (0.5, 0.6): [11, 12, 5, 4, 13, 6, 14, 15, 10, 16, 3, 2, 1, 0, 7, 9, 17, 8, 30, 26],
                (0.6, 0.8): [5, 4, 3, 2, 1, 0, 11, 6, 12, 16, 15, 10, 13, 14, 26, 17, 9, 7, 8, 25]
            }

        elif 0.3 <= input_age_range < 0.4:
            #輸入年齡30-39
            age_specific_points = {
                (0.0, 0.1): [8, 9, 7, 10, 6, 11, 5, 12, 57, 58, 56, 66, 67, 65, 4, 55, 59, 13, 30, 33],
                (0.1, 0.2): [10, 9, 8, 7, 6, 11, 5, 12, 4, 13, 14, 3, 65, 66, 56, 15, 57, 67, 55, 58],
                (0.2, 0.3): [5, 6, 4, 10, 8, 7, 9, 11, 3, 2, 12, 1, 0, 13, 48, 60, 14, 65, 64, 15],
                (0.4, 0.5): [30, 3, 4, 0, 1, 2, 57, 5, 58, 56, 29, 16, 15, 33, 11, 34, 35, 14, 32, 28],
                (0.5, 0.6): [56, 57, 58, 16, 15, 13, 14, 12, 30, 55, 11, 66, 65, 29, 67, 48, 0, 59, 53, 54],
                (0.6, 0.8): [57, 0, 58, 16, 4, 3, 1, 56, 2, 15, 5, 14, 26, 11, 13, 12, 17, 30, 59, 29]
            }


        elif 0.4 <= input_age_range < 0.5:
            #輸入年齡40-49
            age_specific_points = {
                (0.0, 0.1): [8, 9, 7, 10, 6, 11, 5, 12, 4, 13, 3, 14, 56, 57, 65, 55, 66, 58, 67, 2],
                (0.1, 0.2): [10, 11, 5, 6, 9, 8, 7, 4, 12, 13, 3, 14, 2, 15, 16, 1, 0, 65, 54, 55],
                (0.2, 0.3): [5, 4, 3, 2, 6, 11, 1, 0, 10, 12, 13, 16, 9, 15, 7, 14, 8, 17, 26, 30],
                (0.3, 0.4): [ 30, 3, 4, 0, 1, 2, 57, 5, 58, 56, 29, 16, 15, 33, 11, 34, 35, 14, 32, 28],
                (0.5, 0.6): [56, 30, 16, 53, 63, 52, 55, 65, 57, 15, 62, 51, 29, 66, 14, 33, 58, 32, 13, 31],
                (0.6, 0.8): [15, 26, 16, 57, 58, 14, 56, 17, 0, 13, 12, 1, 4, 3, 11, 5, 25, 52, 2, 50]
            }

        elif 0.5 <= input_age_range < 0.6:
            #輸入年齡50-59
            age_specific_points = {
                (0.0, 0.1): [9, 8, 7, 10, 6, 11, 5, 12, 13, 4, 14, 3, 15, 0, 16, 61, 2, 62, 67, 59],
                (0.1, 0.2): [11, 10, 12, 13, 6, 5, 14, 9, 7, 15, 8, 16, 4, 3, 0, 1, 2, 48, 54, 17],
                (0.2, 0.3): [11, 12, 5, 4, 13, 6, 14, 15, 10, 16, 3, 2, 1, 0, 7, 9, 17, 8, 30, 26],
                (0.3, 0.4): [56, 57, 58, 16, 15, 13, 14, 12, 30, 55, 11, 66, 65, 29, 67, 48, 0, 59, 53, 54],
                (0.4, 0.5): [56, 30, 16, 53, 63, 52, 55, 65, 57, 15, 62, 51, 29, 66, 14, 33, 58, 32, 13, 31],
                (0.6, 0.8): [16, 15, 0, 14, 30, 3, 1, 2, 22, 63, 29, 65, 4, 13, 23, 26, 55, 33, 52, 56]
            }

        else:
            #輸入年齡60+
            age_specific_points = {
                (0.0, 0.1): [9, 10, 8, 7, 6, 11, 5, 12, 4, 13, 3, 14, 2, 16, 15, 0, 1, 65, 66, 67],
                (0.1, 0.2): [11, 5, 10, 6, 4, 12, 13, 3, 9, 16, 14, 15, 8, 7, 2, 1, 0, 54, 26, 17],
                (0.2, 0.3): [5, 4, 3, 2, 1, 0, 11, 6, 12, 16, 15, 10, 13, 14, 26, 17, 9, 7, 8, 25],
                (0.3, 0.4): [57, 0, 58, 16, 4, 3, 1, 56, 2, 15, 5, 14, 26, 11, 13, 12, 17, 30, 59, 29],
                (0.4, 0.5): [15, 26, 16, 57, 58, 14, 56, 17, 0, 13, 12, 1, 4, 3, 11, 5, 25, 52, 2, 50],
                (0.5, 0.6): [16, 15, 0, 14, 30, 3, 1, 2, 22, 63, 29, 65, 4, 13, 23, 26, 55, 33, 52, 56]
            }

        # print(f"input age_range: {input_age_range}")
        # print(f"age_range: {target_age_range}")
        # print(f"Available keys: {age_specific_points.keys()}")


        for k, d in enumerate(dets):
            landmarks = np.array([[p.x, p.y] for p in self.predictor(img, d).parts()])

            # Select specific points based on the provided age range
            if target_age_range in age_specific_points:
                selected_points = [landmarks[i] for i in age_specific_points[target_age_range]]
                # print(f"Shape of selected_points: {np.array(selected_points).shape}")


            else:
                # Default to all landmarks if no specific range is provided
                selected_points = landmarks

            return selected_points

        return []  # Return an empty list if no face detected

'''

#亞洲人男性
'''
#亞洲人男性
class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def detect_landmarks(self, pil_img, input_age_range=None, target_age_range=None):
        img = np.array(pil_img)
        dets = self.detector(img, 1)


        # 判断是否跳过检测
        if input_age_range < 0.1 and target_age_range < 0.1:
            return []
        if 0.1 <= input_age_range < 0.2 and 0.1 <= target_age_range < 0.2:
            return []
        if 0.2 <= input_age_range < 0.3 and 0.2 <= target_age_range < 0.3:
            return []
        if 0.3 <= input_age_range < 0.4 and 0.3 <= target_age_range < 0.4:
            return []
        if 0.4 <= input_age_range < 0.5 and 0.4 <= target_age_range < 0.5:
            return []
        if 0.5 <= input_age_range < 0.6 and 0.5 <= target_age_range < 0.6:
            return []
        if input_age_range >= 0.6 and target_age_range >= 0.6:
            return []

        if target_age_range < 0.1:
            target_age_range = (0.0, 0.1)
        elif 0.1 <= target_age_range < 0.2:
            target_age_range = (0.1, 0.2)
        elif 0.2 <= target_age_range < 0.3:
            target_age_range = (0.2, 0.3)
        elif 0.3 <= target_age_range < 0.4:
            target_age_range = (0.3, 0.4)
        elif 0.4 <= target_age_range < 0.5:
            target_age_range = (0.4, 0.5)
        elif 0.5 <= target_age_range < 0.6:
            target_age_range = (0.5, 0.6)
        else:
            target_age_range = (0.6, 0.8)

        if input_age_range < 0.1:
            #輸入年齡10-
            age_specific_points = {
                (0.1, 0.2): [9, 8, 7, 10, 6, 11, 5, 12, 0, 4, 56, 57, 13, 55, 58, 33, 1, 64, 63, 3],
                (0.2, 0.3): [7, 9, 8, 10, 6, 11, 5, 12, 4, 57, 58, 56, 59, 55, 13, 63, 61, 62, 3, 33],
                (0.3, 0.4): [7, 9, 8, 10, 6, 11, 5, 12, 4, 13, 3, 14, 58, 57, 56, 59, 0, 55, 16, 61],
                (0.4, 0.5): [9, 7, 8, 10, 6, 11, 5, 12, 4, 13, 3, 14, 63, 64, 2, 52, 62, 54, 51, 61],
                (0.5, 0.6): [7, 9, 8, 10, 6, 11, 12, 5, 13, 14, 4, 0, 61, 15, 50, 51, 62, 3, 58, 59],
                (0.6, 0.8): [7, 9, 8, 10, 6, 11, 5, 12, 4, 52, 51, 50, 62, 63, 61, 13, 3, 53, 49, 64]
            }
        elif 0.1 <= input_age_range < 0.2:
            #輸入年齡10-19
            age_specific_points = {
                (0.0, 0.1): [9, 8, 7, 10, 6, 11, 5, 12, 0, 4, 56, 57, 13, 55, 58, 33, 1, 64, 63, 3],
                (0.2, 0.3): [6, 5, 7, 4, 3, 8, 2, 9, 16, 1, 10, 0, 15, 26, 14, 11, 13, 12, 58, 59],
                (0.3, 0.4): [5, 4, 6, 3, 7, 2, 8, 1, 9, 10, 11, 0, 26, 12, 45, 16, 13, 46, 15, 14],
                (0.4, 0.5): [4, 5, 3, 6, 2, 7, 8, 11, 10, 9, 1, 12, 0, 13, 52, 50, 51, 26, 14, 16],
                (0.5, 0.6): [7, 11, 8, 6, 12, 9, 13, 10, 5, 14, 4, 15, 26, 3, 50, 51, 52, 49, 16, 2],
                (0.6, 0.8): [ 5, 6, 4, 3, 7, 2, 8, 50, 52, 51, 9, 10, 26, 11, 1, 53, 49, 0, 45, 54]
            }

        elif 0.2 <= input_age_range < 0.3:
            #輸入年齡20-29
            age_specific_points = {
                (0.0, 0.1): [7, 9, 8, 10, 6, 11, 5, 12, 4, 57, 58, 56, 59, 55, 13, 63, 61, 62, 3, 33],
                (0.1, 0.2): [6, 5, 7, 4, 3, 8, 2, 9, 16, 1, 10, 0, 15, 26, 14, 11, 13, 12, 58, 59],
                (0.3, 0.4): [13, 12, 14, 3, 11, 4, 5, 15, 2, 6, 16, 10, 31, 30, 7, 32, 1, 29, 45, 8],
                (0.4, 0.5): [12, 11, 13, 4, 3, 5, 14, 10, 2, 6, 15, 9, 1, 16, 52, 50, 51, 8, 0, 7],
                (0.5, 0.6): [13, 14, 12, 15, 11, 16, 10, 8, 9, 7, 17, 51, 6, 50, 52, 0, 26, 25, 30, 5],
                (0.6, 0.8): [50, 52, 51, 5, 53, 49, 11, 3, 4, 6, 10, 62, 7, 12, 26, 2, 63, 61, 9, 45]
            }

        elif 0.3 <= input_age_range < 0.4:
            #輸入年齡30-39
            age_specific_points = {
                (0.0, 0.1): [7, 9, 8, 10, 6, 11, 5, 12, 4, 13, 3, 14, 58, 57, 56, 59, 0, 55, 16, 61],
                (0.1, 0.2): [5, 4, 6, 3, 7, 2, 8, 1, 9, 10, 11, 0, 26, 12, 45, 16, 13, 46, 15, 14],
                (0.2, 0.3): [13, 12, 14, 3, 11, 4, 5, 15, 2, 6, 16, 10, 31, 30, 7, 32, 1, 29, 45, 8],
                (0.4, 0.5): [52, 51, 50, 11, 4, 5, 30, 63, 10, 62, 12, 53, 3, 9, 61, 34, 33, 49, 8, 54],
                (0.5, 0.6): [14, 13, 12, 15, 16, 11, 0, 17, 10, 8, 7, 9, 51, 50, 18, 52, 25, 1, 26, 19],
                (0.6, 0.8): [50, 52, 51, 53, 49, 62, 63, 61, 54, 48, 30, 64, 5, 60, 6, 26, 0, 7, 33, 8]
            }


        elif 0.4 <= input_age_range < 0.5:
            #輸入年齡40-49
            age_specific_points = {
                (0.0, 0.1): [9, 7, 8, 10, 6, 11, 5, 12, 4, 13, 3, 14, 63, 64, 2, 52, 62, 54, 51, 61],
                (0.1, 0.2): [4, 5, 3, 6, 2, 7, 8, 11, 10, 9, 1, 12, 0, 13, 52, 50, 51, 26, 14, 16],
                (0.2, 0.3): [12, 11, 13, 4, 3, 5, 14, 10, 2, 6, 15, 9, 1, 16, 52, 50, 51, 8, 0, 7],
                (0.3, 0.4): [52, 51, 50, 11, 4, 5, 30, 63, 10, 62, 12, 53, 3, 9, 61, 34, 33, 49, 8, 54],
                (0.5, 0.6): [ 30, 0, 16, 29, 15, 1, 14, 33, 34, 51, 32, 57, 67, 66, 52, 35, 3, 61, 62, 50],
                (0.6, 0.8): [26, 52, 51, 16, 50, 45, 0, 53, 12, 25, 46, 15, 14, 13, 1, 44, 54, 63, 64, 62]
            }

        elif 0.5 <= input_age_range < 0.6:
            #輸入年齡50-59
            age_specific_points = {
                (0.0, 0.1): [7, 9, 8, 10, 6, 11, 12, 5, 13, 14, 4, 0, 61, 15, 50, 51, 62, 3, 58, 59],
                (0.1, 0.2): [7, 11, 8, 6, 12, 9, 13, 10, 5, 14, 4, 15, 26, 3, 50, 51, 52, 49, 16, 2],
                (0.2, 0.3): [13, 14, 12, 15, 11, 16, 10, 8, 9, 7, 17, 51, 6, 50, 52, 0, 26, 25, 30, 5],
                (0.3, 0.4): [14, 13, 12, 15, 16, 11, 0, 17, 10, 8, 7, 9, 51, 50, 18, 52, 25, 1, 26, 19],
                (0.4, 0.5): [30, 0, 16, 29, 15, 1, 14, 33, 34, 51, 32, 57, 67, 66, 52, 35, 3, 61, 62, 50],
                (0.6, 0.8): [14, 15, 16, 13, 0, 12, 1, 30, 11, 2, 26, 3, 19, 50, 29, 25, 58, 57, 20, 51]
            }

        else:
            #輸入年齡60+
            age_specific_points = {
                (0.0, 0.1): [7, 9, 8, 10, 6, 11, 5, 12, 4, 52, 51, 50, 62, 63, 61, 13, 3, 53, 49, 64],
                (0.1, 0.2): [5, 6, 4, 3, 7, 2, 8, 50, 52, 51, 9, 10, 26, 11, 1, 53, 49, 0, 45, 54],
                (0.2, 0.3): [50, 52, 51, 5, 53, 49, 11, 3, 4, 6, 10, 62, 7, 12, 26, 2, 63, 61, 9, 45],
                (0.3, 0.4): [50, 52, 51, 53, 49, 62, 63, 61, 54, 48, 30, 64, 5, 60, 6, 26, 0, 7, 33, 8],
                (0.4, 0.5): [26, 52, 51, 16, 50, 45, 0, 53, 12, 25, 46, 15, 14, 13, 1, 44, 54, 63, 64, 62],
                (0.5, 0.6): [14, 15, 16, 13, 0, 12, 1, 30, 11, 2, 26, 3, 19, 50, 29, 25, 58, 57, 20, 51]
            }

        # print(f"input age_range: {input_age_range}")
        # print(f"age_range: {target_age_range}")
        # print(f"Available keys: {age_specific_points.keys()}")


        for k, d in enumerate(dets):
            landmarks = np.array([[p.x, p.y] for p in self.predictor(img, d).parts()])

            # Select specific points based on the provided age range
            if target_age_range in age_specific_points:
                selected_points = [landmarks[i] for i in age_specific_points[target_age_range]]
                # print(f"Shape of selected_points: {np.array(selected_points).shape}")


            else:
                # Default to all landmarks if no specific range is provided
                selected_points = landmarks

            return selected_points

        return []  # Return an empty list if no face detected

'''

#亞洲人女性
'''
#亞洲人女性
class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def detect_landmarks(self, pil_img, input_age_range=None, target_age_range=None):
        img = np.array(pil_img)
        dets = self.detector(img, 1)


        # 判断是否跳过检测
        if input_age_range < 0.1 and target_age_range < 0.1:
            return []
        if 0.1 <= input_age_range < 0.2 and 0.1 <= target_age_range < 0.2:
            return []
        if 0.2 <= input_age_range < 0.3 and 0.2 <= target_age_range < 0.3:
            return []
        if 0.3 <= input_age_range < 0.4 and 0.3 <= target_age_range < 0.4:
            return []
        if 0.4 <= input_age_range < 0.5 and 0.4 <= target_age_range < 0.5:
            return []
        if 0.5 <= input_age_range < 0.6 and 0.5 <= target_age_range < 0.6:
            return []
        if input_age_range >= 0.6 and target_age_range >= 0.6:
            return []

        if target_age_range < 0.1:
            target_age_range = (0.0, 0.1)
        elif 0.1 <= target_age_range < 0.2:
            target_age_range = (0.1, 0.2)
        elif 0.2 <= target_age_range < 0.3:
            target_age_range = (0.2, 0.3)
        elif 0.3 <= target_age_range < 0.4:
            target_age_range = (0.3, 0.4)
        elif 0.4 <= target_age_range < 0.5:
            target_age_range = (0.4, 0.5)
        elif 0.5 <= target_age_range < 0.6:
            target_age_range = (0.5, 0.6)
        else:
            target_age_range = (0.6, 0.8)

        if input_age_range < 0.1:
            #輸入年齡10-
            age_specific_points = {
                (0.1, 0.2): [9, 8, 7, 10, 30, 6, 33, 0, 57, 56, 11, 32, 58, 34, 62, 5, 61, 63, 55, 35],
                (0.2, 0.3): [8, 9, 7, 10, 6, 57, 56, 58, 30, 5, 11, 33, 55, 66, 59, 65, 67, 34, 32, 62],
                (0.3, 0.4): [8, 9, 7, 10, 6, 11, 5, 57, 56, 58, 12, 55, 65, 66, 4, 67, 59, 13, 64, 3],
                (0.4, 0.5): [8, 7, 9, 10, 6, 11, 5, 12, 4, 13, 14, 3, 58, 59, 57, 67, 15, 0, 60, 56],
                (0.5, 0.6): [9, 8, 10, 7, 6, 11, 12, 5, 13, 0, 14, 4, 15, 1, 16, 3, 2, 57, 58, 54],
                (0.6, 0.8): [8, 9, 7, 10, 6, 11, 5, 12, 4, 13, 3, 14, 16, 15, 0, 2, 1, 67, 60, 59]
            }
        elif 0.1 <= input_age_range < 0.2:
            #輸入年齡10-19
            age_specific_points = {
                (0.0, 0.1): [9, 8, 7, 10, 30, 6, 33, 0, 57, 56, 11, 32, 58, 34, 62, 5, 61, 63, 55, 35],
                (0.2, 0.3): [6, 7, 5, 8, 4, 9, 3, 2, 1, 10, 57, 0, 58, 56, 66, 59, 65, 67, 55, 16],
                (0.3, 0.4): [5, 6, 4, 7, 8, 9, 10, 3, 11, 2, 12, 1, 0, 55, 65, 56, 13, 66, 57, 58],
                (0.4, 0.5): [5, 6, 4, 7, 3, 11, 8, 10, 12, 9, 2, 13, 1, 14, 0, 15, 48, 30, 16, 60],
                (0.5, 0.6): [11, 6, 5, 12, 10, 7, 13, 8, 4, 9, 14, 15, 16, 3, 30, 29, 2, 1, 28, 31],
                (0.6, 0.8): [5, 4, 6, 3, 11, 7, 10, 2, 12, 8, 9, 1, 0, 13, 14, 15, 16, 26, 31, 30]
            }

        elif 0.2 <= input_age_range < 0.3:
            #輸入年齡20-29
            age_specific_points = {
                (0.0, 0.1): [ 8, 9, 7, 10, 6, 57, 56, 58, 30, 5, 11, 33, 55, 66, 59, 65, 67, 34, 32, 62],
                (0.1, 0.2): [6, 7, 5, 8, 4, 9, 3, 2, 1, 10, 57, 0, 58, 56, 66, 59, 65, 67, 55, 16],
                (0.3, 0.4): [11, 5, 10, 12, 6, 4, 9, 7, 8, 13, 3, 14, 54, 64, 2, 53, 15, 1, 60, 48],
                (0.4, 0.5): [5, 12, 11, 6, 4, 13, 7, 10, 14, 3, 30, 8, 31, 15, 9, 16, 29, 2, 32, 49],
                (0.5, 0.6): [12, 11, 13, 14, 16, 15, 30, 10, 29, 31, 32, 6, 5, 17, 0, 1, 33, 26, 28, 35],
                (0.6, 0.8): [5, 11, 12, 4, 13, 6, 3, 14, 10, 2, 15, 0, 1, 16, 7, 31, 26, 8, 9, 30]
            }

        elif 0.3 <= input_age_range < 0.4:
            #輸入年齡30-39
            age_specific_points = {
                (0.0, 0.1): [8, 9, 7, 10, 6, 11, 5, 57, 56, 58, 12, 55, 65, 66, 4, 67, 59, 13, 64, 3],
                (0.1, 0.2): [5, 6, 4, 7, 8, 9, 10, 3, 11, 2, 12, 1, 0, 55, 65, 56, 13, 66, 57, 58],
                (0.2, 0.3): [11, 5, 10, 12, 6, 4, 9, 7, 8, 13, 3, 14, 54, 64, 2, 53, 15, 1, 60, 48],
                (0.4, 0.5): [30, 29, 31, 32, 57, 33, 58, 56, 5, 28, 12, 6, 66, 67, 13, 34, 65, 7, 4, 35],
                (0.5, 0.6): [30, 56, 57, 54, 1, 64, 58, 29, 2, 55, 0, 16, 65, 53, 66, 3, 15, 48, 14, 22],
                (0.6, 0.8): [57, 15, 56, 14, 13, 58, 16, 12, 5, 0, 30, 11, 1, 4, 29, 31, 3, 2, 26, 32]
            }


        elif 0.4 <= input_age_range < 0.5:
            #輸入年齡40-49
            age_specific_points = {
                (0.0, 0.1): [8, 7, 9, 10, 6, 11, 5, 12, 4, 13, 14, 3, 58, 59, 57, 67, 15, 0, 60, 56],
                (0.1, 0.2): [ 5, 6, 4, 7, 3, 11, 8, 10, 12, 9, 2, 13, 1, 14, 0, 15, 48, 30, 16, 60],
                (0.2, 0.3): [ 5, 12, 11, 6, 4, 13, 7, 10, 14, 3, 30, 8, 31, 15, 9, 16, 29, 2, 32, 49],
                (0.3, 0.4): [30, 29, 31, 32, 57, 33, 58, 56, 5, 28, 12, 6, 66, 67, 13, 34, 65, 7, 4, 35],
                (0.5, 0.6): [3, 2, 4, 48, 60, 1, 5, 0, 49, 6, 59, 54, 64, 7, 17, 8, 53, 36, 50, 18],
                (0.6, 0.8): [15, 16, 14, 26, 0, 1, 13, 52, 53, 2, 50, 51, 12, 3, 49, 4, 54, 48, 58, 57]
            }

        elif 0.5 <= input_age_range < 0.6:
            #輸入年齡50-59
            age_specific_points = {
                (0.0, 0.1): [9, 8, 10, 7, 6, 11, 12, 5, 13, 0, 14, 4, 15, 1, 16, 3, 2, 57, 58, 54],
                (0.1, 0.2): [11, 6, 5, 12, 10, 7, 13, 8, 4, 9, 14, 15, 16, 3, 30, 29, 2, 1, 28, 31],
                (0.2, 0.3): [12, 11, 13, 14, 16, 15, 30, 10, 29, 31, 32, 6, 5, 17, 0, 1, 33, 26, 28, 35],
                (0.3, 0.4): [30, 56, 57, 54, 1, 64, 58, 29, 2, 55, 0, 16, 65, 53, 66, 3, 15, 48, 14, 22],
                (0.4, 0.5): [3, 2, 4, 48, 60, 1, 5, 0, 49, 6, 59, 54, 64, 7, 17, 8, 53, 36, 50, 18],
                (0.6, 0.8): [3, 2, 4, 1, 0, 5, 16, 6, 15, 60, 48, 18, 14, 17, 7, 8, 19, 13, 20, 10]
            }

        else:
            #輸入年齡60+
            age_specific_points = {
                (0.0, 0.1): [8, 9, 7, 10, 6, 11, 5, 12, 4, 13, 3, 14, 16, 15, 0, 2, 1, 67, 60, 59],
                (0.1, 0.2): [5, 4, 6, 3, 11, 7, 10, 2, 12, 8, 9, 1, 0, 13, 14, 15, 16, 26, 31, 30],
                (0.2, 0.3): [5, 11, 12, 4, 13, 6, 3, 14, 10, 2, 15, 0, 1, 16, 7, 31, 26, 8, 9, 30],
                (0.3, 0.4): [57, 15, 56, 14, 13, 58, 16, 12, 5, 0, 30, 11, 1, 4, 29, 31, 3, 2, 26, 32],
                (0.4, 0.5): [15, 16, 14, 26, 0, 1, 13, 52, 53, 2, 50, 51, 12, 3, 49, 4, 54, 48, 58, 57],
                (0.5, 0.6): [3, 2, 4, 1, 0, 5, 16, 6, 15, 60, 48, 18, 14, 17, 7, 8, 19, 13, 20, 10]
            }

        # print(f"input age_range: {input_age_range}")
        # print(f"age_range: {target_age_range}")
        # print(f"Available keys: {age_specific_points.keys()}")


        for k, d in enumerate(dets):
            landmarks = np.array([[p.x, p.y] for p in self.predictor(img, d).parts()])

            # Select specific points based on the provided age range
            if target_age_range in age_specific_points:
                selected_points = [landmarks[i] for i in age_specific_points[target_age_range]]
                # print(f"Shape of selected_points: {np.array(selected_points).shape}")


            else:
                # Default to all landmarks if no specific range is provided
                selected_points = landmarks

            return selected_points

        return []  # Return an empty list if no face detected


'''

#黑人男
'''
class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def detect_landmarks(self, pil_img, input_age_range=None, target_age_range=None):
        img = np.array(pil_img)
        dets = self.detector(img, 1)


        # 判断是否跳过检测
        if input_age_range < 0.2 and target_age_range < 0.2:
            return []
        if 0.2 <= input_age_range < 0.3 and 0.2 <= target_age_range < 0.3:
            return []
        if 0.3 <= input_age_range < 0.4 and 0.3 <= target_age_range < 0.4:
            return []
        if 0.4 <= input_age_range < 0.5 and 0.4 <= target_age_range < 0.5:
            return []
        if 0.5 <= input_age_range < 0.6 and 0.5 <= target_age_range < 0.6:
            return []
        if 0.6 <= input_age_range < 0.7 and 0.6 <= target_age_range < 0.7:
            return []
        if input_age_range >= 0.7 and target_age_range >= 0.7:
            return []

        if target_age_range < 0.2:
            target_age_range = (0.0, 0.2)
        elif 0.2 <= target_age_range < 0.3:
            target_age_range = (0.2, 0.3)
        elif 0.3 <= target_age_range < 0.4:
            target_age_range = (0.3, 0.4)
        elif 0.4 <= target_age_range < 0.5:
            target_age_range = (0.4, 0.5)
        elif 0.5 <= target_age_range < 0.6:
            target_age_range = (0.5, 0.6)
        elif 0.6 <= target_age_range < 0.7:
            target_age_range = (0.6, 0.7)
        else:
            target_age_range = (0.7, 0.9)

        if input_age_range < 0.2:
            #輸入年齡20-
            age_specific_points = {
                (0.2, 0.3): [7, 9, 8, 6, 10, 5, 11, 4, 3, 12, 2, 56, 57, 55, 58, 59, 16, 66, 65, 60],
                (0.3, 0.4): [6, 7, 9, 8, 5, 10, 4, 11, 3, 12, 2, 16, 1, 13, 0, 15, 14, 26, 48, 60],
                (0.4, 0.5): [6, 7, 5, 9, 8, 10, 4, 3, 11, 2, 12, 1, 13, 16, 0, 14, 15, 54, 64, 26],
                (0.5, 0.6): [6, 7, 5, 8, 9, 4, 10, 3, 2, 11, 1, 12, 16, 0, 13, 15, 14, 65, 56, 55],
                (0.6, 0.7): [6, 5, 7, 9, 8, 4, 10, 3, 11, 16, 2, 1, 15, 12, 0, 14, 13, 26, 52, 17],
                (0.7, 0.9): [7, 6, 8, 9, 5, 10, 4, 11, 3, 12, 2, 16, 14, 50, 51, 15, 1, 0, 13, 61]
            }

        elif 0.2 <= input_age_range < 0.3:
            #輸入年齡20-29
            age_specific_points = {
                (0.0, 0.2): [7, 9, 8, 6, 10, 5, 11, 4, 3, 12, 2, 56, 57, 55, 58, 59, 16, 66, 65, 60],
                (0.3, 0.4): [30, 4, 56, 57, 58, 3, 5, 29, 35, 55, 31, 34, 16, 32, 2, 33, 59, 65, 61, 63],
                (0.4, 0.5): [ 3, 4, 5, 2, 1, 6, 11, 12, 0, 13, 14, 16, 10, 15, 7, 9, 30, 8, 35, 57],
                (0.5, 0.6): [3, 4, 2, 5, 1, 0, 6, 16, 30, 7, 8, 15, 29, 35, 9, 34, 14, 21, 12, 13],
                (0.6, 0.7): [ 16, 15, 3, 2, 4, 1, 0, 14, 5, 26, 17, 13, 58, 12, 57, 18, 56, 25, 35, 6],
                (0.7, 0.9): [ 0, 17, 14, 16, 18, 15, 1, 49, 48, 50, 2, 19, 60, 6, 51, 12, 54, 13, 5, 63]
            }

        elif 0.3 <= input_age_range < 0.4:
            #輸入年齡30-39
            age_specific_points = {
                (0.0, 0.2): [ 6, 7, 9, 8, 5, 10, 4, 11, 3, 12, 2, 16, 1, 13, 0, 15, 14, 26, 48, 60],
                (0.2, 0.3): [ 30, 4, 56, 57, 58, 3, 5, 29, 35, 55, 31, 34, 16, 32, 2, 33, 59, 65, 61, 63],
                (0.4, 0.5): [3, 4, 2, 5, 1, 6, 9, 8, 7, 0, 10, 11, 13, 12, 62, 54, 63, 61, 50, 14],
                (0.5, 0.6): [3, 2, 4, 1, 0, 5, 6, 7, 8, 21, 67, 66, 20, 57, 58, 65, 56, 30, 9, 29],
                (0.6, 0.7): [16, 15, 0, 14, 1, 2, 3, 48, 18, 26, 17, 19, 50, 4, 25, 49, 60, 37, 13, 20],
                (0.7, 0.9): [49, 50, 48, 60, 51, 61, 63, 62, 53, 52, 54, 19, 64, 18, 0, 30, 20, 17, 1, 2]
            }


        elif 0.4 <= input_age_range < 0.5:
            #輸入年齡40-49
            age_specific_points = {
                (0.0, 0.2): [6, 7, 5, 9, 8, 10, 4, 3, 11, 2, 12, 1, 13, 16, 0, 14, 15, 54, 64, 26],
                (0.2, 0.3): [3, 4, 5, 2, 1, 6, 11, 12, 0, 13, 14, 16, 10, 15, 7, 9, 30, 8, 35, 57],
                (0.3, 0.4): [3, 4, 2, 5, 1, 6, 9, 8, 7, 0, 10, 11, 13, 12, 62, 54, 63, 61, 50, 14],
                (0.5, 0.6): [0, 21, 67, 58, 66, 14, 57, 12, 13, 20, 56, 1, 15, 11, 59, 2, 16, 65, 3, 22],
                (0.6, 0.7): [16, 15, 14, 13, 0, 37, 1, 38, 12, 18, 19, 24, 26, 17, 11, 2, 20, 23, 44, 21],
                (0.7, 0.9): [49, 50, 51, 53, 63, 61, 52, 54, 60, 48, 0, 62, 1, 30, 2, 19, 64, 20, 3, 18]
            }

        elif 0.5 <= input_age_range < 0.6:
            #輸入年齡50-59
            age_specific_points = {
                (0.0, 0.2): [6, 7, 5, 8, 9, 4, 10, 3, 2, 11, 1, 12, 16, 0, 13, 15, 14, 65, 56, 55],
                (0.2, 0.3): [3, 4, 2, 5, 1, 0, 6, 16, 30, 7, 8, 15, 29, 35, 9, 34, 14, 21, 12, 13],
                (0.3, 0.4): [3, 2, 4, 1, 0, 5, 6, 7, 8, 21, 67, 66, 20, 57, 58, 65, 56, 30, 9, 29],
                (0.4, 0.5): [ 0, 21, 67, 58, 66, 14, 57, 12, 13, 20, 56, 1, 15, 11, 59, 2, 16, 65, 3, 22],
                (0.6, 0.7): [0, 1, 2, 56, 57, 58, 3, 7, 8, 16, 9, 59, 6, 4, 44, 55, 26, 45, 66, 15],
                (0.7, 0.9): [0, 1, 2, 30, 63, 3, 51, 53, 50, 49, 61, 62, 52, 54, 29, 4, 48, 60, 64, 33]
            }
        elif 0.6 <= input_age_range < 0.7:
            #輸入年齡60-69
            age_specific_points = {
                (0.0, 0.2): [6, 5, 7, 9, 8, 4, 10, 3, 11, 16, 2, 1, 15, 12, 0, 14, 13, 26, 52, 17],
                (0.2, 0.3): [16, 15, 3, 2, 4, 1, 0, 14, 5, 26, 17, 13, 58, 12, 57, 18, 56, 25, 35, 6],
                (0.3, 0.4): [16, 15, 0, 14, 1, 2, 3, 48, 18, 26, 17, 19, 50, 4, 25, 49, 60, 37, 13, 20],
                (0.4, 0.5): [16, 15, 14, 13, 0, 37, 1, 38, 12, 18, 19, 24, 26, 17, 11, 2, 20, 23, 44, 21],
                (0.5, 0.6): [0, 1, 2, 56, 57, 58, 3, 7, 8, 16, 9, 59, 6, 4, 44, 55, 26, 45, 66, 15],
                (0.7, 0.9): [30, 49, 63, 50, 61, 51, 62, 53, 52, 60, 54, 48, 29, 64, 58, 55, 34, 57, 33, 56]
            }

        else:
            #輸入年齡70+
            age_specific_points = {
                (0.0, 0.2): [7, 6, 8, 9, 5, 10, 4, 11, 3, 12, 2, 16, 14, 50, 51, 15, 1, 0, 13, 61],
                (0.2, 0.3): [ 0, 17, 14, 16, 18, 15, 1, 49, 48, 50, 2, 19, 60, 6, 51, 12, 54, 13, 5, 63],
                (0.3, 0.4): [49, 50, 48, 60, 51, 61, 63, 62, 53, 52, 54, 19, 64, 18, 0, 30, 20, 17, 1, 2],
                (0.4, 0.5): [49, 50, 51, 53, 63, 61, 52, 54, 60, 48, 0, 62, 1, 30, 2, 19, 64, 20, 3, 18],
                (0.5, 0.6): [0, 1, 2, 30, 63, 3, 51, 53, 50, 49, 61, 62, 52, 54, 29, 4, 48, 60, 64, 33],
                (0.6, 0.7): [30, 49, 63, 50, 61, 51, 62, 53, 52, 60, 54, 48, 29, 64, 58, 55, 34, 57, 33, 56]
            }

        # print(f"input age_range: {input_age_range}")
        # print(f"age_range: {target_age_range}")
        # print(f"Available keys: {age_specific_points.keys()}")


        for k, d in enumerate(dets):
            landmarks = np.array([[p.x, p.y] for p in self.predictor(img, d).parts()])

            # Select specific points based on the provided age range
            if target_age_range in age_specific_points:
                selected_points = [landmarks[i] for i in age_specific_points[target_age_range]]
                # print(f"Shape of selected_points: {np.array(selected_points).shape}")


            else:
                # Default to all landmarks if no specific range is provided
                selected_points = landmarks

            return selected_points

        return []  # Return an empty list if no face detected
'''

#黑人女
class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def detect_landmarks(self, pil_img, input_age_range=None, target_age_range=None):
        img = np.array(pil_img)
        dets = self.detector(img, 1)


        # 判断是否跳过检测
        if input_age_range < 0.2 and target_age_range < 0.2:
            return []
        if 0.2 <= input_age_range < 0.3 and 0.2 <= target_age_range < 0.3:
            return []
        if 0.3 <= input_age_range < 0.4 and 0.3 <= target_age_range < 0.4:
            return []
        if 0.4 <= input_age_range < 0.5 and 0.4 <= target_age_range < 0.5:
            return []
        if 0.5 <= input_age_range < 0.6 and 0.5 <= target_age_range < 0.6:
            return []
        if 0.6 <= input_age_range < 0.7 and 0.6 <= target_age_range < 0.7:
            return []
        if input_age_range >= 0.7 and target_age_range >= 0.7:
            return []

        if target_age_range < 0.2:
            target_age_range = (0.0, 0.2)
        elif 0.2 <= target_age_range < 0.3:
            target_age_range = (0.2, 0.3)
        elif 0.3 <= target_age_range < 0.4:
            target_age_range = (0.3, 0.4)
        elif 0.4 <= target_age_range < 0.5:
            target_age_range = (0.4, 0.5)
        elif 0.5 <= target_age_range < 0.6:
            target_age_range = (0.5, 0.6)
        elif 0.6 <= target_age_range < 0.7:
            target_age_range = (0.6, 0.7)
        else:
            target_age_range = (0.7, 0.9)

        if input_age_range < 0.2:
            #輸入年齡20-
            age_specific_points = {
                (0.2, 0.3): [8, 9, 7, 10, 6, 11, 5, 12, 13, 4, 0, 58, 14, 57, 56, 3, 59, 15, 2, 1],
                (0.3, 0.4): [8, 9, 10, 7, 6, 11, 5, 12, 13, 4, 14, 15, 16, 3, 0, 2, 1, 57, 58, 56],
                (0.4, 0.5): [6, 10, 5, 11, 7, 8, 9, 16, 0, 12, 4, 15, 13, 1, 14, 2, 3, 26, 25, 59],
                (0.5, 0.6): [11, 10, 16, 12, 15, 9, 0, 6, 13, 8, 7, 14, 5, 1, 4, 2, 26, 17, 3, 25],
                (0.6, 0.7): [11, 12, 0, 10, 13, 16, 1, 14, 15, 5, 6, 9, 2, 4, 8, 7, 3, 17, 54, 18],
                (0.7, 0.9): [16, 0, 1, 5, 4, 15, 2, 3, 14, 11, 6, 12, 13, 10, 7, 9, 8, 26, 54, 48]
            }

        elif 0.2 <= input_age_range < 0.3:
            #輸入年齡20-29
            age_specific_points = {
                (0.0, 0.2): [8, 9, 7, 10, 6, 11, 5, 12, 13, 4, 0, 58, 14, 57, 56, 3, 59, 15, 2, 1],
                (0.3, 0.4): [11, 10, 9, 8, 6, 5, 12, 7, 4, 13, 14, 16, 15, 3, 30, 2, 1, 29, 0, 28],
                (0.4, 0.5): [16, 15, 0, 5, 1, 26, 4, 17, 14, 2, 3, 25, 6, 18, 13, 11, 12, 24, 30, 19],
                (0.5, 0.6): [17, 26, 16, 57, 58, 56, 15, 18, 25, 59, 55, 67, 31, 11, 32, 30, 0, 66, 14, 65],
                (0.6, 0.7): [0, 16, 1, 15, 12, 11, 2, 14, 13, 17, 3, 4, 10, 26, 5, 18, 58, 36, 57, 48],
                (0.7, 0.9): [16, 0, 1, 15, 2, 3, 4, 14, 5, 13, 57, 12, 58, 17, 11, 26, 56, 6, 30, 10]
            }

        elif 0.3 <= input_age_range < 0.4:
            #輸入年齡30-39
            age_specific_points = {
                (0.0, 0.2): [8, 9, 10, 7, 6, 11, 5, 12, 13, 4, 14, 15, 16, 3, 0, 2, 1, 57, 58, 56],
                (0.2, 0.3): [ 11, 10, 9, 8, 6, 5, 12, 7, 4, 13, 14, 16, 15, 3, 30, 2, 1, 29, 0, 28],
                (0.4, 0.5): [16, 26, 25, 15, 17, 24, 0, 18, 23, 19, 22, 14, 57, 1, 56, 58, 20, 45, 44, 21],
                (0.5, 0.6): [57, 58, 56, 17, 26, 59, 67, 55, 18, 66, 25, 7, 65, 8, 9, 16, 19, 24, 6, 60],
                (0.6, 0.7): [0, 1, 2, 16, 17, 15, 14, 3, 18, 12, 13, 58, 26, 57, 56, 11, 4, 36, 19, 25],
                (0.7, 0.9): [16, 0, 1, 15, 2, 14, 3, 4, 57, 58, 5, 13, 56, 17, 26, 12, 18, 66, 59, 67]
            }


        elif 0.4 <= input_age_range < 0.5:
            #輸入年齡40-49
            age_specific_points = {
                (0.0, 0.2): [6, 10, 5, 11, 7, 8, 9, 16, 0, 12, 4, 15, 13, 1, 14, 2, 3, 26, 25, 59],
                (0.2, 0.3): [16, 15, 0, 5, 1, 26, 4, 17, 14, 2, 3, 25, 6, 18, 13, 11, 12, 24, 30, 19],
                (0.3, 0.4): [3, 2, 16, 26, 25, 15, 17, 24, 0, 18, 23, 19, 22, 14, 57, 1, 56, 58, 20, 45, 44, 21],
                (0.5, 0.6): [4, 1, 2, 3, 0, 5, 26, 6, 25, 58, 59, 7, 64, 57, 8, 56, 67, 54, 55, 9],
                (0.6, 0.7): [ 0, 1, 2, 12, 48, 60, 13, 11, 16, 15, 3, 14, 17, 49, 36, 4, 41, 59, 37, 10],
                (0.7, 0.9): [1, 0, 2, 16, 3, 15, 4, 14, 58, 57, 5, 49, 50, 51, 13, 52, 56, 30, 61, 21]
            }

        elif 0.5 <= input_age_range < 0.6:
            #輸入年齡50-59
            age_specific_points = {
                (0.0, 0.2): [11, 10, 16, 12, 15, 9, 0, 6, 13, 8, 7, 14, 5, 1, 4, 2, 26, 17, 3, 25],
                (0.2, 0.3): [17, 26, 16, 57, 58, 56, 15, 18, 25, 59, 55, 67, 31, 11, 32, 30, 0, 66, 14, 65],
                (0.3, 0.4): [57, 58, 56, 17, 26, 59, 67, 55, 18, 66, 25, 7, 65, 8, 9, 16, 19, 24, 6, 60],
                (0.4, 0.5): [4, 1, 2, 3, 0, 5, 26, 6, 25, 58, 59, 7, 64, 57, 8, 56, 67, 54, 55, 9],
                (0.6, 0.7): [48, 4, 2, 1, 60, 3, 5, 0, 12, 13, 6, 14, 11, 7, 49, 54, 59, 15, 17, 64],
                (0.7, 0.9): [1, 2, 0, 3, 4, 5, 16, 15, 14, 13, 6, 12, 48, 49, 60, 54, 50, 11, 64, 51]
            }
        elif 0.6 <= input_age_range < 0.7:
            #輸入年齡60-69
            age_specific_points = {
                (0.0, 0.2): [11, 12, 0, 10, 13, 16, 1, 14, 15, 5, 6, 9, 2, 4, 8, 7, 3, 17, 54, 18],
                (0.2, 0.3): [0, 16, 1, 15, 12, 11, 2, 14, 13, 17, 3, 4, 10, 26, 5, 18, 58, 36, 57, 48],
                (0.3, 0.4): [0, 1, 2, 16, 17, 15, 14, 3, 18, 12, 13, 58, 26, 57, 56, 11, 4, 36, 19, 25],
                (0.4, 0.5): [ 0, 1, 2, 12, 48, 60, 13, 11, 16, 15, 3, 14, 17, 49, 36, 4, 41, 59, 37, 10],
                (0.5, 0.6): [48, 4, 2, 1, 60, 3, 5, 0, 12, 13, 6, 14, 11, 7, 49, 54, 59, 15, 17, 64],
                (0.7, 0.9): [0, 1, 2, 3, 16, 15, 4, 14, 5, 13, 17, 18, 12, 19, 6, 20, 57, 58, 41, 11]
            }

        else:
            #輸入年齡70+
            age_specific_points = {
                (0.0, 0.2): [16, 0, 1, 5, 4, 15, 2, 3, 14, 11, 6, 12, 13, 10, 7, 9, 8, 26, 54, 48],
                (0.2, 0.3): [16, 0, 1, 15, 2, 3, 4, 14, 5, 13, 57, 12, 58, 17, 11, 26, 56, 6, 30, 10],
                (0.3, 0.4): [16, 0, 1, 15, 2, 14, 3, 4, 57, 58, 5, 13, 56, 17, 26, 12, 18, 66, 59, 67],
                (0.4, 0.5): [1, 0, 2, 16, 3, 15, 4, 14, 58, 57, 5, 49, 50, 51, 13, 52, 56, 30, 61, 21],
                (0.5, 0.6): [ 1, 2, 0, 3, 4, 5, 16, 15, 14, 13, 6, 12, 48, 49, 60, 54, 50, 11, 64, 51],
                (0.6, 0.7): [0, 1, 2, 3, 16, 15, 4, 14, 5, 13, 17, 18, 12, 19, 6, 20, 57, 58, 41, 11]
            }

        # print(f"input age_range: {input_age_range}")
        # print(f"age_range: {target_age_range}")
        # print(f"Available keys: {age_specific_points.keys()}")


        for k, d in enumerate(dets):
            landmarks = np.array([[p.x, p.y] for p in self.predictor(img, d).parts()])

            # Select specific points based on the provided age range
            if target_age_range in age_specific_points:
                selected_points = [landmarks[i] for i in age_specific_points[target_age_range]]
                # print(f"Shape of selected_points: {np.array(selected_points).shape}")


            else:
                # Default to all landmarks if no specific range is provided
                selected_points = landmarks

            return selected_points

        return []  # Return an empty list if no face detected



'''

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

    def detect_landmarks(self, pil_img):
        img = np.array(pil_img)
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
        #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        return [point4, point5, point6, point7, point8, point9, point10, point11, point12]
        
        
if __name__ == "__main__":

    #predictor_path = "/home/tony/SAM-master/shape_predictor_68_face_landmarks.dat"
    png_path = "D:/SAM-master/celebAMask-HQ/test_img/1.jpg"

    landmark_detector = FaceLandmarkDetector(opts.face_landmarks_path)
    #landmark_detector.detect_landmarks(png_path)
    x,y = landmark_detector.detect_landmarks(png_path)
    print(x)
    print(y)

'''

