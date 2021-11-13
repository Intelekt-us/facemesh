import os
import time
from datetime import datetime
from enum import Enum
from math import asin, atan2, pi

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf


class HeadPosition:
    def __init__(self, direction, roll_vector=None):
        self.direction = direction
        self.roll_vector = roll_vector
        self.yaw = self._yaw(direction)
        self.pitch = self._pitch(direction)
        if roll_vector is not None:
            self.roll = self._roll(roll_vector)

    def is_in_region(self, region_boundary):
        return region_boundary.up > self.pitch > region_boundary.down \
               and region_boundary.left < self.yaw < region_boundary.right

    def is_in_region_as_vector(self, region_boundary, attention_center_vector):
        return self.pitch - self._pitch(attention_center_vector) < region_boundary.up and self.pitch - self._pitch(
            attention_center_vector) > region_boundary.down \
               and self.yaw - self._yaw(attention_center_vector) > region_boundary.left and self.yaw - self._yaw(
            attention_center_vector) < region_boundary.right

    @classmethod
    def _yaw(cls, direction):
        return 90 + atan2(direction[2], direction[0]) * 180 / pi

    @classmethod
    def _pitch(cls, direction):
        return -90 + atan2(direction[2], direction[1]) * 180 / pi * -1

    @classmethod
    def _roll(cls, roll_vector):
        return np.abs(
            asin(np.sum(np.dot(roll_vector, np.array([0., 1., 0.]))) / np.sqrt(np.sum(roll_vector ** 2))) * 180 / pi)

    @classmethod
    def from_landmarks_list(cls, landmarks):
        left_ear = landmarks[234]
        right_ear = landmarks[454]
        chin = landmarks[151]
        forehead = landmarks[9]
        horizontal_vector = right_ear - left_ear
        vertical_verctor = forehead - chin

        normal = np.cross(vertical_verctor, horizontal_vector)
        normal = cls._normalize(normal)

        return cls(normal, cls._normalize(horizontal_vector))

    @classmethod
    def _normalize(cls, vector):
        return vector / np.sqrt(np.sum(vector ** 2))


class Visualization:
    def __init__(self, default_font_size=0.5, show_position=True, show_fps=True, show_region=True,
                 show_attention_center=True, show_talking=True, show_sleep=True, show_data_feed=True,
                 show_iris_coordinates=True, show_screen_distance=True, show_iris_position=True):
        self.vc = cv2.VideoCapture(0)
        ret, frame = self.vc.read()
        if not ret:
            raise Exception("Could not initialize camera")
        else:
            self.image_size = (frame.shape[1], frame.shape[0])
        self.show_position = show_position
        self.show_attention_center = show_attention_center
        self.show_fps = show_fps
        self.show_region = show_region
        self.show_talking = show_talking
        self.show_sleep = show_sleep
        self.show_data_feed = show_data_feed
        self.show_iris_coordinates = show_iris_coordinates
        self.show_screen_distance = show_screen_distance
        self.show_iris_position = show_iris_position
        self.last_time = time.time()
        self.fps = 0
        self.default_font = cv2.FONT_HERSHEY_SIMPLEX
        self.default_font_size = default_font_size
        self.iris_position = [0., 0.]

    def read_image(self):
        success, self.image = self.vc.read()
        self._update_fps()
        return (success, self.image)

    def _update_fps(self):
        current_time = time.time()
        self.fps = int(1 / (current_time - self.last_time))
        self.last_time = current_time

    def display_position(self, head_position):
        cv2.putText(self.image, f"YAW: {head_position.yaw:.1f}", (5, 30), self.default_font, 0.5, (255, 0, 0), 2)
        cv2.putText(self.image, f"PITCH: {head_position.pitch:.1f}", (5, 50), self.default_font, 0.5, (255, 0, 0), 2)
        if head_position.roll is not None:
            cv2.putText(self.image, f"ROLL: {head_position.roll:.1f}", (5, 70), self.default_font, 0.5, (255, 0, 0), 2)

    def display_direction(self, head_position):
        cv2.putText(self.image,
                    f"[{head_position.direction[0]:.2f}, {head_position.direction[1]:.2f}, {head_position.direction[2]:.2f}] (HEAD POSITION)",
                    (5, 200), self.default_font, 0.5, (0, 0, 255), 2)

    def display_attention_center(self, attention_center):
        cv2.putText(self.image,
                    f"[{attention_center.vector[0]:.2f}, {attention_center.vector[1]:.2f}, {attention_center.vector[2]:.2f}] (ATTENTION CENTER)",
                    (5, 220), self.default_font, 0.5, (0, 0, 255), 2)

    def display_fps(self):
        cv2.putText(self.image, f"FPS: {self.fps}", (5, 100), self.default_font, 0.5, (0, 255, 0), 2)

    def display_detected_region(self, region):
        cv2.putText(self.image, f'ENGAGEMENT LEVEL: {region.name}', (5, 120), self.default_font, 0.5, region.value, 2)

    def display_talking(self, talk_checker):
        is_talking = talk_checker.is_talking()
        if is_talking:
            cv2.putText(self.image, "SPEAKING", (5, 170), self.default_font, 0.5, (255, 255, 0), 2)
        else:
            cv2.putText(self.image, "LISTENING", (5, 170), self.default_font, 0.5, (255, 255, 0), 2)

    def display_sleepiness(self, sleepiness):
        is_sleeping = sleepiness.is_sleeping()
        if is_sleeping:
            cv2.putText(self.image, f"AWAKE", (5, 185), self.default_font, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(self.image, f"AWAKE ", (5, 185), self.default_font, 0.5, (255, 255, 255), 2)

    def display_storage(self, storage):
        cv2.putText(self.image, f"TOTAL DATA STORED BY INTELLECTUS:", (5, 260), self.default_font, 0.5, (255, 0, 0), 2)
        for i in range(len(storage.data)):
            cv2.putText(self.image, str(storage.data[i]), (5, 280 + 20 * i), self.default_font, 0.5,
                        storage.data[i].region.value, 2)

    def display_iris_coordinates(self, iris_coordinates):
        for eye in iris_coordinates:
            for point in eye:
                cv2.circle(self.image, (int(point[0]), int(point[1])), 0, (255, 0, 0), 3)

    def display_screen_distance(self, screen_distance):
        cv2.putText(self.image,
                    f"DISTANCE FROM THE SCREEN: {int(screen_distance * 0.0393700787)}in ({int(screen_distance / 10.)}cm)",
                    (5, 135), self.default_font, 0.5, (0, 255, 0), 2)

    def display_iris_position(self, iris_position):
        self.iris_position[0] = self.iris_position[0] * 0.9 + iris_position[0] * 0.1
        self.iris_position[1] = self.iris_position[1] * 0.9 + iris_position[1] * 0.1
        cv2.putText(self.image, f"EYEBALL DIRECTION: [{self.iris_position[0]:.2f},{self.iris_position[1]:.2f}]",
                    (5, 150), self.default_font, 0.8, (0, 255, 0), 2)

    def show(self, head_position=None, region=None, attention_center=None, talk_checker=None, sleepiness=None,
             storage=None, iris_coordinates=None, screen_distance=None, iris_position=None):
        if self.show_iris_coordinates and iris_coordinates:
            self.display_iris_coordinates(iris_coordinates)
        cv2.putText(self.image, f"TOTAL DATA PROCESSED LOCALLY: ", (5, 15), self.default_font, 0.5, (255, 0, 0), 2)
        if self.show_position and head_position:
            self.display_position(head_position)
            self.display_direction(head_position)
        if self.show_attention_center and attention_center:
            self.display_attention_center(attention_center)
        if self.show_fps:
            self.display_fps()
        if self.show_region and region:
            self.display_detected_region(region)
        if self.show_talking and talk_checker:
            self.display_talking(talk_checker)
        if self.show_sleep and sleepiness:
            self.display_sleepiness(sleepiness)
        if self.show_data_feed and storage:
            self.display_storage(storage)
        if self.show_screen_distance and screen_distance:
            self.display_screen_distance(screen_distance)
        if self.show_iris_position and iris_position is not None:
            self.display_iris_position(iris_position)
        # elif not iris_position:
        # self.display_iris_position("Neither iris detected")

        cv2.imshow("vid", self.image)

    def is_return_key_pressed(self):
        return cv2.waitKey(1) & 0xFF == ord('q')


class FaceMesh:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=detection_confidence,
                                                         min_tracking_confidence=tracking_confidence)

    def get_face_landmarks(self, image):
        reverted_colors_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(reverted_colors_image)
        try:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
        except:
            return None
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        return landmarks


class Regions(Enum):
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    NOT_PRESENT = (0, 0, 0)


class RegionBoundary:
    def __init__(self, up, right, down, left):
        self.up = up
        self.down = down
        self.right = right
        self.left = left


class AttentionCenter:
    def __init__(self, EMA_alpha):
        self.vector = np.array([0, 0, -1])
        self.EMA_alpha = EMA_alpha

    def UpdateAttention_EMA(self, detected_region):
        if detected_region == Regions.YELLOW or detected_region == Regions.GREEN:
            self.vector = AttentionCenter._normalize(
                head_position.direction * self.EMA_alpha + (1 - self.EMA_alpha) * self.vector)

    @classmethod
    def _normalize(cls, vector):
        return vector / np.sqrt(np.sum(vector ** 2))


class Attention:
    def __init__(self, green_region_boundary, yellow_region_boundary,
                 eye_green_region_boundary, eye_yellow_region_boundary, ema_factor=0.,
                 attention_center=np.array([0, 0, -1])):
        self.green_region_boundary = green_region_boundary  # solution using angles
        self.yellow_region_boundary = yellow_region_boundary
        self.eye_green_region_boundary = eye_green_region_boundary
        self.eye_yellow_region_boundary = eye_yellow_region_boundary
        self.head_position = None
        self.detected_region = None
        self.attention_center = attention_center
        self.iris_position = [0., 0.]
        self.extendreg = np.array([0., 0., 0., 0.])
        self.ema_factor = ema_factor
        self.iris_Eaverage = [0., 0.]
        self.head_screen_distance = 0

    def update_green_boundary(self):
        if MA_detected_region == Regions.YELLOW:
            if IrisGreenBoundary.up < self.extendreg[0] < IrisYellowBoundary.up:
                IrisGreenBoundary.up = self.extendreg[0]
            if IrisGreenBoundary.right < self.extendreg[1] < IrisYellowBoundary.right:
                IrisGreenBoundary.right = self.extendreg[1]
            if IrisGreenBoundary.down > self.extendreg[2] > IrisYellowBoundary.down:
                IrisGreenBoundary.down = self.extendreg[2]
            if IrisGreenBoundary.left > self.extendreg[3] > IrisYellowBoundary.left:
                IrisGreenBoundary.left = self.extendreg[3]
            if (self.head_screen_distance / 10) > 50:
                IrisGreenBoundary.up = (-0.0017 / 2) * (self.head_screen_distance / 10) + IrisGreenBoundary.up
                IrisGreenBoundary.right = (-0.0021 / 2) * (self.head_screen_distance / 10) + IrisGreenBoundary.right
                IrisGreenBoundary.down = (0.0027 / 2) * (self.head_screen_distance / 10) + IrisGreenBoundary.down
                IrisGreenBoundary.left = (0.0021 / 2) * (self.head_screen_distance / 10) + IrisGreenBoundary.left
            else:
                IrisGreenBoundary.up = 0.17
                IrisGreenBoundary.right = 0.17
                IrisGreenBoundary.down = -0.25
                IrisGreenBoundary.left = -0.17

    def update_yellow_boundary(self):
        if (self.head_screen_distance / 10) > 50:
            IrisYellowBoundary.up = (-0.0017 / 1.5) * (self.head_screen_distance / 10) + 0.305
            IrisYellowBoundary.right = (-0.0021 / 1.5) * (self.head_screen_distance / 10) + 0.319
            IrisYellowBoundary.down = (0.0027 / 1.5) * (self.head_screen_distance / 10) - 0.395
            IrisYellowBoundary.left = (0.0021 / 1.5) * (self.head_screen_distance / 10) - 0.319
        else:
            IrisYellowBoundary.up = 0.3
            IrisYellowBoundary.right = 0.3
            IrisYellowBoundary.down = -0.35
            IrisYellowBoundary.left = -0.3

    def calculate_iris_Eaverage(self):
        if MA_detected_region == Regions.GREEN or MA_detected_region == Regions.YELLOW:
            self.iris_Eaverage[0] = self.iris_Eaverage[0] * (1 - self.ema_factor) + \
                                    self.iris_position[0] * self.ema_factor
            self.iris_Eaverage[1] = self.iris_Eaverage[1] * (1 - self.ema_factor) + \
                                    self.iris_position[1] * self.ema_factor
        return self.iris_Eaverage

    def calculate_extreg(self):
        if MA_detected_region == Regions.GREEN or MA_detected_region == Regions.YELLOW:
            self.extendreg[0] = max(self.extendreg[0], self.calculate_iris_Eaverage()[1])
            self.extendreg[1] = max(self.extendreg[1], self.calculate_iris_Eaverage()[0])
            self.extendreg[2] = min(self.extendreg[2], self.calculate_iris_Eaverage()[1])
            self.extendreg[3] = min(self.extendreg[3], self.calculate_iris_Eaverage()[0])
        return self.extendreg

    def set_eye_green_region_boundary(self, boundary):
        self.eye_green_region_boundary = boundary

    def iris_position_is_in_region(self, region_boundary):
        return region_boundary.up > self.iris_position[1] > region_boundary.down \
               and region_boundary.left < self.iris_position[0] < region_boundary.right

    def update_head_position(self, new_position):
        self.head_position = new_position

    def update_attention_center(self, new_attention):
        self.attention_center = new_attention

    def get_detected_region_from_saved_position_as_vector(self):
        self.detected_region = None
        self.detected_region = None
        if not self.head_position:
            self.detected_region = Regions.NOT_PRESENT
        elif not (self.head_position.is_in_region(self.yellow_region_boundary)
                  and self.iris_position_is_in_region(self.eye_yellow_region_boundary)):
            self.detected_region = Regions.RED
        else:
            if self.head_position.is_in_region_as_vector(self.green_region_boundary, attention_center.vector) \
                    and self.iris_position_is_in_region(self.eye_green_region_boundary):
                self.detected_region = Regions.GREEN
            else:
                self.detected_region = Regions.YELLOW
        return self.detected_region


class handRaise:
    def __init__(self, min_visibility):
        self.if_raise = False
        self.min_visibility = min_visibility
        self.results = None
        self.head_landmark = None
        self.right_hand_landmark = None
        self.left_hand_landmark = None
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0)

    def get_pose_landmarks(self, image):
        self.results = self.pose.process(image)
        self.head_landmark = self.results.pose_landmarks.landmark[9]
        self.right_hand_landmark = self.results.pose_landmarks.landmark[16]
        self.left_hand_landmark = self.results.pose_landmarks.landmark[15]

    def if_landmark_up(self, hand_landmark):
        if hand_landmark.visibility > self.min_visibility and hand_landmark.y < self.head_landmark.y:
            return True
        else:
            return False

    def check_hand_raised(self):
        if (self.if_landmark_up(self.right_hand_landmark) or self.if_landmark_up(self.left_hand_landmark)):
            self.if_raise = True
            print("Hand is raised!")
        else:
            self.if_raise = False
            print("-")
        print(self.if_raise)
        return self.if_raise

    def draw_pose(self, image):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            self.results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


class TalkChecker:
    def __init__(self, delay=20):
        self.delay = delay
        self.not_talking_time = delay

    def update(self, lips_movement):
        if lips_movement is not None:
            is_talking = lips_movement.is_mouth_open()
            if is_talking:
                self.not_talking_time = 0
            else:
                self.not_talking_time += 1

    def is_talking(self):
        return self.not_talking_time < self.delay


class LipsMovement:
    def __init__(self, upper_lip_position, lower_lip_position, left_mouth_corner, right_mouth_corner,
                 openness_threshold):
        self.upper_lip_position = upper_lip_position
        self.lower_lip_position = lower_lip_position
        self.left_mouth_corner = left_mouth_corner
        self.right_mouth_corner = right_mouth_corner
        self.openness_threshold = openness_threshold

    @classmethod
    def from_landmarks_list(cls, landmarks, openness_threshold=0.07):
        upper_lip_position = landmarks[13]
        lower_lip_position = landmarks[14]
        left_mouth_corner = landmarks[78]
        right_mouth_corner = landmarks[308]
        return cls(upper_lip_position, lower_lip_position, left_mouth_corner, right_mouth_corner, openness_threshold)

    def is_mouth_open(self):
        return self.calculate_openness() > self.openness_threshold

    def calculate_openness(self):
        return np.linalg.norm(self.upper_lip_position - self.lower_lip_position) / np.linalg.norm(
            self.left_mouth_corner - self.right_mouth_corner)


class Sleepiness:
    def __init__(self, threshold, alpha):
        self.threshold = threshold
        self.MA_openness = 1.
        self.alpha = alpha

    def update(self, eyelids_movement):
        if eyelids_movement:
            self.MA_openness = self.MA_openness * (1 - self.alpha) + eyelids_movement.mean_openness() * self.alpha

    def is_sleeping(self):
        return self.MA_openness < self.threshold


class EyesMovement:
    def __init__(self, left_eye, right_eye, openness_threshold=0.05):
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.openness_threshold = openness_threshold

    def are_open(self):
        return self.mean_openness() > self.openness_threshold

    def mean_openness(self):
        return (self.left_eye.calculate_openness() + self.right_eye.calculate_openness()) / 2

    @classmethod
    def from_landmarks_list(cls, landmarks):
        left_eye_upper_lid = landmarks[159]
        left_eye_lower_lid = landmarks[145]
        left_eye_left_corner = landmarks[33]
        left_eye_right_corner = landmarks[133]
        right_eye_upper_lid = landmarks[386]
        right_eye_lower_lid = landmarks[374]
        right_eye_left_corner = landmarks[362]
        right_eye_right_corner = landmarks[263]
        left_eye = SingleEyeMovement(left_eye_upper_lid, left_eye_lower_lid, left_eye_left_corner,
                                     left_eye_right_corner)
        right_eye = SingleEyeMovement(right_eye_upper_lid, right_eye_lower_lid, right_eye_left_corner,
                                      right_eye_right_corner)
        return cls(left_eye, right_eye)


class SingleEyeMovement:
    def __init__(self, upper_lid, lower_lid, left_corner, right_corner):
        self.upper_lid = upper_lid
        self.lower_lid = lower_lid
        self.left_corner = left_corner
        self.right_corner = right_corner

    def calculate_openness(self):
        return np.linalg.norm(self.upper_lid - self.lower_lid) / np.linalg.norm(self.left_corner - self.right_corner)


class StorageEntry:
    def __init__(self, region, previous_region):
        self.previous_region = previous_region
        self.region = region
        self.time = datetime.now()

    def __str__(self):
        if self.region == Regions.GREEN:
            if self.previous_region == Regions.NOT_PRESENT:
                r = "REJOINED"
            else:
                r = "FULLY ENGAGED"
        elif self.region == Regions.YELLOW:
            if self.previous_region == Regions.GREEN:
                r = "ENGAGEMENT DROPPED OFF"
            elif self.previous_region == Regions.YELLOW:
                r = "PRESENT"
            elif self.previous_region == Regions.RED:
                r = "ENGAGEMENT RESTORED"
            elif self.previous_region == Regions.NOT_PRESENT:
                r = "REJOINED"
        elif self.region == Regions.RED:
            if self.previous_region == Regions.NOT_PRESENT:
                r = "REJOINED"
            else:
                r = "ENGAGEMENT LOST"
        elif self.region == Regions.NOT_PRESENT:
            if self.previous_region == Regions.GREEN or self.previous_region == Regions.YELLOW:
                r = "LEFT MEETING"
            else:
                r = "ABSENT"

        return self.time.strftime("%H:%M:%S.%f - %b %d %Y - ") + r


class Storage:
    def __init__(self, saved_frames, frames_delay=10):
        self.saved_frames = saved_frames
        self.data = [StorageEntry(Regions.GREEN, Regions.GREEN)]
        self.delay = frames_delay
        self.timer = 1

    def add(self, entry):
        if self.timer < self.delay:
            self.timer += 1
        else:
            self.timer = 1
            if len(self.data) > self.saved_frames:
                self.data.pop(0)
            self.data.append(StorageEntry(entry, self.data[-1].region))


class ROI:
    def __init__(self, center, max_size):
        self.center = center
        self.max_size = max_size

    @classmethod
    def from_eye_landmarks(cls, eye_landmarks, scale=2.0):
        x_min = np.min(eye_landmarks[:, 0])
        x_max = np.max(eye_landmarks[:, 0])
        y_min = np.min(eye_landmarks[:, 1])
        y_max = np.max(eye_landmarks[:, 1])
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        width = (x_max - x_min) * scale
        height = (y_max - y_min) * scale
        max_size = max(width, height)
        return cls((x_center, y_center), max_size)

    def crop_image(self, image):
        x_min = int(self.center[0] - self.max_size / 2)
        y_min = int(self.center[1] - self.max_size / 2)
        x_max = int(self.center[0] + self.max_size / 2)
        y_max = int(self.center[1] + self.max_size / 2)
        return image[y_min:y_max, x_min:x_max]

    def get_in_frame_position(self):
        return (self.center[0] - self.max_size / 2, self.center[1] - self.max_size / 2)


class IrisDetector:
    IMAGE_SIZE = (64, 64)

    def __init__(self, tensorflow_model_path):
        self.tensorflow_model_path = tensorflow_model_path
        self._initialize_tensorflow_model()

    def landmarks_from_image(self, image, is_right_eye=True):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            return None
        low_res_image = cv2.resize(image, IrisDetector.IMAGE_SIZE, interpolation=cv2.INTER_AREA) / 127.5 - 1.0
        if is_right_eye:
            low_res_image = cv2.flip(low_res_image, 1)
        outputs = self._tflite_inference(low_res_image)
        iris_landmarks = np.reshape(outputs[1], (5, 3))
        iris_landmarks /= IrisDetector.IMAGE_SIZE[0]
        if is_right_eye:
            iris_landmarks[:, 0] = 1 - iris_landmarks[:, 0]

        iris_landmarks[:, 0] *= image.shape[1]
        iris_landmarks[:, 1] *= image.shape[0]
        return iris_landmarks

    def _initialize_tensorflow_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.tensorflow_model_path)
        self.interpreter.allocate_tensors()

    def _tflite_inference(self, inputs):
        inputs = np.array([inputs])
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        for inp, inp_det in zip(inputs, input_details):
            self.interpreter.set_tensor(inp_det["index"], np.array(inp[None, ...], dtype=np.float32))

        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(out["index"]) for out in output_details]

        return outputs


class IrisAnalysis:
    def __init__(self, image_size, human_iris_size_in_mm):
        self.image_size = image_size
        self.focal_length = image_size[0]
        self.human_iris_size_in_mm = human_iris_size_in_mm

    def eyes_relative_position(self, face, left_iris, right_iris):
        if right_iris is None and left_iris is None:
            return None
        elif right_iris is None:
            return self._single_eye_relative_position(face, left_iris, is_right_eye=False)
        elif left_iris is None:
            return self._single_eye_relative_position(face, right_iris, is_right_eye=True)
        else:
            return (self._single_eye_relative_position(face, right_iris, is_right_eye=True)
                    + self._single_eye_relative_position(face, left_iris, is_right_eye=False)) / 2

    def _single_eye_relative_position(self, face, iris, is_right_eye=False):
        if not is_right_eye:
            right_corner = face[33, :2]
            left_corner = face[133, :2]
            upper_eyelid = face[159, :2]
            lower_eyelid = face[145, :2]
        else:
            left_corner = face[263, :2]
            right_corner = face[362, :2]
            upper_eyelid = face[386, :2]
            lower_eyelid = face[374, :2]
        iris_center = iris[0, :2]
        left_center_distance = np.linalg.norm(left_corner - iris_center)
        right_center_distance = np.linalg.norm(right_corner - iris_center)
        upper_center_distance = np.linalg.norm(upper_eyelid - iris_center)
        lower_center_distance = np.linalg.norm(lower_eyelid - iris_center)
        vertical_position = 2 * lower_center_distance / (upper_center_distance + lower_center_distance) - 1
        horizontal_position = 2 * left_center_distance / (left_center_distance + right_center_distance) - 1
        return np.array([horizontal_position, vertical_position])

    def distance_from_irises(self, left_iris, right_iris):
        if left_iris is None and right_iris is None:
            return None
        elif left_iris is None:
            return self._distance_from_single_iris(right_iris)
        elif right_iris is None:
            self._distance_from_single_iris(left_iris)
        else:
            return (self._distance_from_single_iris(left_iris) + self._distance_from_single_iris(right_iris)) / 2

    def _distance_from_single_iris(self, iris):

        origin = np.array(self.image_size) / 2.0
        center_landmark = iris[0, :2]
        x = np.sqrt(np.sum((origin - center_landmark) ** 2) + self.focal_length ** 2)
        return self.human_iris_size_in_mm / self._iris_size(iris) * x

    def _iris_size(self, iris):
        size = (np.linalg.norm(iris[1, :2] - iris[3, :2]) + np.linalg.norm(iris[2, :2] - iris[4, :2])) / 2
        return size


class FinalStats:
    def __init__(self, dataframe):
        meeting_time = FinalStatsCreator.get_meeting_time(dataframe)
        self.listening_time = meeting_time.listening_time
        self.speaking_time = meeting_time.speaking_time
        self.absent_time = FinalStatsCreator.get_absent_time(dataframe)
        self.total_time = FinalStatsCreator.get_total_time(dataframe)
        self.screen_distance = FinalStatsCreator.get_average_screen_distance(dataframe)

    def save_report(self, filename):
        with open(filename, 'w+') as file:
            file.write(self.get_report_text())

    def get_report_text(self):
        return f"Time spent speaking: {self.speaking_time * 100:.1f}% of meeting\n" \
               + f"Time spent listening: {self.listening_time * 100:.1f}% of meeting\n" \
               + f"Time you were absent: {self.absent_time * 100:.1f}% of meeting\n" \
               + f"Total meeting time: {self.total_time}s\n" \
               + f"Average screen distance: {self.screen_distance / 10:.0f}cm\n"


class MeetingTime:
    def __init__(self, speaking_time, listening_time):
        self.listening_time = listening_time
        self.speaking_time = speaking_time


class FinalStatsCreator:
    @staticmethod
    def get_meeting_time(dataframe):
        speaking_time = 0
        listening_time = 0
        current_time = dataframe.time[0]
        last_time = dataframe.time[0]
        for index, row in dataframe.iterrows():
            if row.is_mouth_open:
                speaking_time += (current_time - last_time).total_seconds()
            elif row.attention_region == Regions.YELLOW or row.attention_region == Regions.GREEN:
                listening_time += (current_time - last_time).total_seconds()
            last_time = current_time
            current_time = row.time

        total_time = FinalStatsCreator.get_total_time(dataframe)

        listening_time = listening_time / total_time
        speaking_time = speaking_time / total_time

        return MeetingTime(speaking_time, listening_time)

    @staticmethod
    def get_absent_time(dataframe):
        absent_time = 0
        current_time = dataframe.time[0]
        last_time = dataframe.time[0]
        for index, row in dataframe.iterrows():
            if row.attention_region == Regions.NOT_PRESENT:
                absent_time += (current_time - last_time).total_seconds()
            last_time = current_time
            current_time = row.time
        total_time = FinalStatsCreator.get_total_time(dataframe)
        return absent_time / total_time

    @staticmethod
    def get_average_screen_distance(dataframe):
        return dataframe.screen_distance.mean()

    @staticmethod
    def get_total_time(dataframe):
        return (dataframe.iloc[-1].time - dataframe.time[0]).total_seconds()


if __name__ == "__main__":
    face_mesh = FaceMesh()
    hand_raise = handRaise(min_visibility=0.8)
    IrisYellowBoundary = RegionBoundary(0.3, 0.3, -0.35, -0.3)
    IrisGreenBoundary = RegionBoundary(0.17, 0.17, -0.25, -0.17)
    visualization = Visualization()
    attention = Attention(
        RegionBoundary(30, 30, -8, -30),  # green region
        RegionBoundary(35, 50, -10, -50),  # yellow region
        IrisGreenBoundary,  # up,right,down,left; green eye region
        IrisYellowBoundary,  # yellow eye region
        ema_factor=0.0075)
    sleepiness = Sleepiness(0.25, 0.05)
    talk_checker = TalkChecker()
    storage = Storage(10)
    LEFT_EYE_LANDMARKS_ID = np.array([33, 133])
    RIGHT_EYE_LANDMARKS_ID = np.array([362, 263])
    iris_detector = IrisDetector(tensorflow_model_path="models/iris_landmark.tflite")
    iris_analysis = IrisAnalysis(image_size=visualization.image_size, human_iris_size_in_mm=11.8)
    attention_center: AttentionCenter = AttentionCenter(EMA_alpha=0.005)
    MA_detected_region = None
    start_time = time.time()

    df = pd.DataFrame(columns=['time', 'yaw', 'pitch', 'roll', 'eyes_position', 'attention_vector', 'attention_region',
                               'screen_distance',
                               'mouth_openness', 'eyes_openness', 'is_mouth_open'])

    while True:

        success, image = visualization.read_image()
        if not success:
            break

        landmarks = face_mesh.get_face_landmarks(image)
        hand_raise.get_pose_landmarks(image)
        if landmarks is None:
            head_position = None
            lips_movement = None
            eyelids_movement = None
            left_eye_landmarks = None
            right_eye_landmarks = None
            left_iris_landmarks = None
            right_iris_landmarks = None
            iris_coordinates = None
            eyes_position = None
            head_screen_distance = None
        else:
            head_position = HeadPosition.from_landmarks_list(landmarks)
            landmarks[:, 0] *= image.shape[1]
            landmarks[:, 1] *= image.shape[0]
            lips_movement = LipsMovement.from_landmarks_list(landmarks)
            eyelids_movement = EyesMovement.from_landmarks_list(landmarks)
            left_eye_landmarks = landmarks[LEFT_EYE_LANDMARKS_ID]
            right_eye_landmarks = landmarks[RIGHT_EYE_LANDMARKS_ID]
            left_roi = ROI.from_eye_landmarks(left_eye_landmarks)
            right_roi = ROI.from_eye_landmarks(right_eye_landmarks)
            left_eye_image = left_roi.crop_image(image)
            right_eye_image = right_roi.crop_image(image)
            left_iris_landmarks = iris_detector.landmarks_from_image(left_eye_image, is_right_eye=False)
            right_iris_landmarks = iris_detector.landmarks_from_image(right_eye_image, is_right_eye=True)
            left_in_frame_position = left_roi.get_in_frame_position()
            right_in_frame_position = right_roi.get_in_frame_position()

            try:
                right_iris_landmarks[:, 0] += right_in_frame_position[0]
                right_iris_landmarks[:, 1] += right_in_frame_position[1]
                left_iris_landmarks[:, 0] += left_in_frame_position[0]
                left_iris_landmarks[:, 1] += left_in_frame_position[1]
                iris_coordinates = [right_iris_landmarks, left_iris_landmarks]
            except:
                eyes_position = None
                head_screen_distance = None
                iris_coordinates = None
            eyes_position = iris_analysis.eyes_relative_position(landmarks, left_iris_landmarks, right_iris_landmarks)
            attention.iris_position = eyes_position
            head_screen_distance = iris_analysis.distance_from_irises(left_iris_landmarks, right_iris_landmarks)
            attention.head_screen_distance = head_screen_distance
            attention_center.UpdateAttention_EMA(MA_detected_region)

        attention.update_head_position(head_position)
        attention.update_attention_center(attention_center)
        hand_raise.check_hand_raised()
        sleepiness.update(eyelids_movement)
        talk_checker.update(lips_movement)

        MA_detected_region = attention.get_detected_region_from_saved_position_as_vector()
        storage.add(MA_detected_region)
        if not MA_detected_region == Regions.RED:
            attention.calculate_iris_Eaverage()
            attention.calculate_extreg()
            attention.update_green_boundary()
            attention.update_yellow_boundary()

        if head_position is not None:
            df = df.append({'time': datetime.now(), 'yaw': head_position.yaw,
                            'pitch': head_position.pitch, 'roll': head_position.roll,
                            'eyes_position': eyes_position, 'attention_vector': head_position.direction,
                            'attention_region': attention.detected_region,
                            'screen_distance': head_screen_distance,
                            'mouth_openness': lips_movement.calculate_openness(),
                            'eyes_openness': eyelids_movement.mean_openness(),
                            'is_mouth_open': talk_checker.is_talking()}, ignore_index=True)
        else:
            df = df.append({'time': datetime.now(), 'yaw': np.nan, 'pitch': np.nan,
                            'roll': np.nan, 'eyes_position': np.nan, 'attention_vector': np.nan,
                            'attention_region': Regions.NOT_PRESENT,
                            'screen_distance': np.nan, 'mouth_openness': np.nan, 'eyes_openness': np.nan,
                            'is_mouth_open': np.nan},
                           ignore_index=True)
        visualization.show(head_position=head_position, region=MA_detected_region, attention_center=attention_center,
                           talk_checker=talk_checker, sleepiness=sleepiness, storage=storage,
                           iris_coordinates=iris_coordinates, screen_distance=head_screen_distance,
                           iris_position=eyes_position)
        hand_raise.draw_pose(image)
        if visualization.is_return_key_pressed():
            break
    if not os.path.exists('./data/'):
        os.mkdir('data')

    # noinspection PyTypeChecker
    filename = datetime.now().isoformat(timespec='seconds')
    df.to_csv(f"./data/{filename}.csv")
    final_report = FinalStats(df)
    final_report.save_report(f"./data/{filename}_report.txt")
