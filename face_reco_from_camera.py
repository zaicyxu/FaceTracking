# Real-time face detection and recognition

import dlib
import cv2
import pandas as pd
import numpy as np
import os
import time
import logging
from PIL import Image, ImageDraw, ImageFont

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class Face_Recognizer:
    def __init__(self):
        self.face_feature_known_list = []  # Save the features of faces in database
        self.face_name_known_list = []  # Save the name of faces in database

        self.current_frame_face_cnt = 0  # Counter for faces in current frame
        self.current_frame_face_feature_list = []  # Features of faces in current frame
        self.current_frame_face_name_list = []  # Names of faces in current frame
        self.current_frame_face_name_position_list = []  # Positions of faces in current frame

        # Update FPS
        self.fps = 0
        self.frame_start_time = 0
        self.frame_cnt = 0

        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)

    # Read known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_feature_known_list.append(features_someone_arr)
                self.face_name_known_list.append("Person_" + str(i + 1))
            logging.info("Faces in Database：%d", len(self.face_feature_known_list))
            return 1

        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    # Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        # Write names under rectangle
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_name_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_face_name_position_list[i], text=self.current_frame_face_name_list[i],
                      font=self.font_chinese,
                      fill=(255, 255, 0))
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd

    # Show names in chinese
    def show_chinese_name(self):
        # Default known name: person_1, person_2, person_3
        if self.current_frame_face_cnt >= 1:
            # Modify names in face_name_known_list to chinese name
            self.face_name_known_list[0] = '张三'.encode('utf-8').decode()
            # self.face_name_known_list[1] = '张四'.encode('utf-8').decode()

    # Face detection and recognition from input video stream
    def process(self, stream):
        # 1.Read known faces from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame %d starts", self.frame_cnt)
                flag, img_rd = stream.read()
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(1)
                # Press 'q' to quit
                if kk == ord('q'):
                    break
                else:
                    self.draw_note(img_rd)
                    self.current_frame_face_feature_list = []
                    self.current_frame_face_cnt = 0
                    self.current_frame_face_name_position_list = []
                    self.current_frame_face_name_list = []

                    # 2.Face detected in current frame
                    if len(faces) != 0:

                        # 3.Compute the face descriptors for faces in current frame
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))

                        # 4.Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("For face %d in camera:", k + 1)
                            # Set the default names of faces with "unknown"
                            self.current_frame_face_name_list.append("unknown")
                            # Positions of faces captured
                            self.current_frame_face_name_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 5.For every faces detected, compare the faces in the database
                            current_frame_e_distance_list = []
                            for i in range(len(self.face_feature_known_list)):
                                if str(self.face_feature_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_feature_known_list[i])
                                    logging.debug("  With person %s, the e-distence is %f", str(i + 1), e_distance_tmp)
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    current_frame_e_distance_list.append(999999999)

                            # 6. Find the one with minimum e-distance
                            similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                            logging.debug("Minimum e-distance with %s: %f",
                                          self.face_name_known_list[similar_person_num],
                                          min(current_frame_e_distance_list))

                            if min(current_frame_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.debug("Face recognition result: %s",
                                              self.face_name_known_list[similar_person_num])
                            else:
                                logging.debug("Face recognition result: Unknown person")
                            logging.debug("\n")
                            # Draw rectangle
                            for kk, d in enumerate(faces):
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (255, 255, 255), 2)

                        self.current_frame_face_cnt = len(faces)

                        # 7. Modify name if needed
                        self.show_chinese_name()

                        # 8. Draw name
                        img_with_name = self.draw_name(img_rd)
                    else:
                        img_with_name = img_rd
                logging.debug("Faces in camera now: %s", self.current_frame_face_name_list)
                cv2.imshow("camera", img_with_name)

                # 9. Update stream FPS
                self.update_fps()
                logging.debug("Frame ends\n\n")

    # OpenCV 调用摄像头并进行 process
    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)  # Get video stream from camera
        cap.set(3, 480)  # 640   x480
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()

def main():
    # logging.basicConfig(level=logging.DEBUG)
    # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()

if __name__ == '__main__':
    main()