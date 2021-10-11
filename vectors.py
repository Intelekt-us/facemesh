import mediapipe as mp
import numpy as np
import cv2
from math import atan2, pi, asin
from enum import Enum
import time
import pandas as pd
from os import path, mkdir
from datetime import datetime
from iris_test import get_iris_coordinates


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
        return region_boundary.up > self.pitch - self._pitch(attention_center_vector) > region_boundary.down \
               and region_boundary.left < self.yaw - self._yaw(attention_center_vector) < region_boundary.right

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
        horizontal_vector = np.array([right_ear.x - left_ear.x, right_ear.y - left_ear.y, right_ear.z - left_ear.z])
        vertical_verctor = np.array([forehead.x - chin.x, forehead.y - chin.y, forehead.z - chin.z])

        normal = np.cross(vertical_verctor, horizontal_vector)
        normal = cls._normalize(normal)

        return cls(normal, cls._normalize(horizontal_vector))

    @classmethod
    def _normalize(cls, vector):
        return vector / np.sqrt(np.sum(vector ** 2))


class Visualization:
    def __init__(self, default_font_size=0.5, show_position=True, show_fps=True, show_region=True,
                 show_attention_center=True, show_talking=True, show_sleep=True, show_data_feed=True,
                 show_iris_coordinates=True):
        self.vc = cv2.VideoCapture(0)
        self.show_position = show_position
        self.show_attention_center = show_attention_center
        self.show_fps = show_fps
        self.show_region = show_region
        self.show_talking = show_talking
        self.show_sleep = show_sleep
        self.show_data_feed = show_data_feed
        self.show_iris_coordinates = show_iris_coordinates
        self.last_time = time.time()
        self.fps = 0
        self.default_font = cv2.FONT_HERSHEY_SIMPLEX
        self.default_font_size = default_font_size

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
            cv2.putText(self.image, "SPEAKING", (5, 160), self.default_font, 0.5, (255, 255, 0), 2)
        else:
            cv2.putText(self.image, "LISTENING", (5, 160), self.default_font, 0.5, (255, 255, 0), 2)

    def display_sleepiness(self, sleepiness):
        is_sleeping = sleepiness.is_sleeping()
        if is_sleeping:
            cv2.putText(self.image, f"NAPPING", (5, 180), self.default_font, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(self.image, f"AWAKE", (5, 180), self.default_font, 0.5, (255, 255, 255), 2)

    def display_storage(self, storage):
        cv2.putText(self.image, f"TOTAL DATA STORED BY INTELLECTUS:", (5, 260), self.default_font, 0.5, (255, 0, 0), 2)
        for i in range(len(storage.data)):
            cv2.putText(self.image, str(storage.data[i]), (5, 280 + 20 * i), self.default_font, 0.5,
                        storage.data[i].region.value, 2)

    def display_iris_coordinates(self, iris_coordinates):
        for eye in iris_coordinates:
            for point in eye:
                cv2.circle(self.image, (int(point[0] * self.image.shape[1]), int(point[1] * self.image.shape[0])), 0,
                           (0, 0, 255), 3)

    def show(self, head_position=None, region=None, attention_center=None, talk_checker=None, sleepiness=None,
             storage=None, iris_coordinates=None):
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
        if self.show_iris_coordinates and iris_coordinates:
            self.display_iris_coordinates(iris_coordinates)

        cv2.imshow("vid", self.image)

    @staticmethod
    def is_return_key_pressed():
        return cv2.waitKey(1) & 0xFF == ord('q')


class FaceMesh:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=detection_confidence,
                                                         min_tracking_confidence=tracking_confidence)

    def get_face_landmarks(self, image):
        reverted_colors_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(reverted_colors_image)
        try:
            return mesh_results.multi_face_landmarks[0].landmark
        except:
            return None


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
    def __init__(self, yellow_region_boundary, EMA_alpha=0.01):
        self.vector = np.array([0, 0, -1])
        self.EMA_alpha = EMA_alpha
        self.yellow_region_boundary = yellow_region_boundary

    def UpdateAttention_dummy(self):
        self.vector = np.array([0, 0, -1])

    def UpdateAttention_EMA(self, detected_region):
        if detected_region == Regions.YELLOW or detected_region == Regions.GREEN:
            self.vector = AttentionCenter._normalize(
                head_position.direction * self.EMA_alpha + (1 - self.EMA_alpha) * self.vector)

    @classmethod
    def _normalize(cls, vector):
        return vector / np.sqrt(np.sum(vector ** 2))


class Attention:
    def __init__(self, green_region_boundary, yellow_region_boundary, attention_center=np.array([0, 0, -1])):
        self.green_region_boundary = green_region_boundary  # solution using angles
        self.yellow_region_boundary = yellow_region_boundary
        self.head_position = None
        self.detected_region = None
        self.attention_center = attention_center

    def set_green_region_boundary(self, boundary):
        self.green_region_boundary = boundary

    def update_head_position(self, new_position):
        self.head_position = new_position

    def update_attention_center(self, new_attention):
        self.attention_center = new_attention

    def get_detected_region_from_saved_position_as_vector(self):
        self.detected_region = None
        self.detected_region = None
        if not self.head_position:
            self.detected_region = Regions.NOT_PRESENT
        elif not self.head_position.is_in_region(self.yellow_region_boundary):
            self.detected_region = Regions.RED
        else:
            if self.head_position.is_in_region_as_vector(self.green_region_boundary, attention_center.vector):
                self.detected_region = Regions.GREEN
            else:
                self.detected_region = Regions.YELLOW
        return self.detected_region

    def get_detected_region_from_saved_position(self):
        self.detected_region = None
        if not self.head_position:
            self.detected_region = Regions.NOT_PRESENT
        else:
            if self.head_position.is_in_region(self.green_region_boundary):
                self.detected_region = Regions.GREEN
            elif self.head_position.is_in_region(self.yellow_region_boundary):
                self.detected_region = Regions.YELLOW
            else:
                self.detected_region = Regions.RED
        return self.detected_region


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
        upper_lip_position = LipsMovement._landmark_to_vector(landmarks[13])
        lower_lip_position = LipsMovement._landmark_to_vector(landmarks[14])
        left_mouth_corner = LipsMovement._landmark_to_vector(landmarks[78])
        right_mouth_corner = LipsMovement._landmark_to_vector(landmarks[308])
        return cls(upper_lip_position, lower_lip_position, left_mouth_corner, right_mouth_corner, openness_threshold)

    def is_mouth_open(self):
        return self._calculate_openness() > self.openness_threshold

    def _calculate_openness(self):
        return np.linalg.norm(self.upper_lip_position - self.lower_lip_position) / np.linalg.norm(
            self.left_mouth_corner - self.right_mouth_corner)

    @classmethod
    def _landmark_to_vector(cls, landmark):
        return np.array([landmark.x, landmark.y, landmark.z])


class Sleepiness:
    def __init__(self, threshold, alpha):
        self.threshold = threshold
        self.MA_openness = 1.
        self.alpha = alpha

    def update(self, eyes_movement):
        if eyes_movement:
            self.MA_openness = self.MA_openness * (1 - self.alpha) + eyes_movement.mean_openness() * self.alpha

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
        left_eye_upper_lid = EyesMovement._landmark_to_vector(landmarks[159])
        left_eye_lower_lid = EyesMovement._landmark_to_vector(landmarks[145])
        left_eye_left_corner = EyesMovement._landmark_to_vector(landmarks[33])
        left_eye_right_corner = EyesMovement._landmark_to_vector(landmarks[133])
        right_eye_upper_lid = EyesMovement._landmark_to_vector(landmarks[386])
        right_eye_lower_lid = EyesMovement._landmark_to_vector(landmarks[374])
        right_eye_left_corner = EyesMovement._landmark_to_vector(landmarks[362])
        right_eye_right_corner = EyesMovement._landmark_to_vector(landmarks[263])
        left_eye = SingleEyeMovement(left_eye_upper_lid, left_eye_lower_lid, left_eye_left_corner,
                                     left_eye_right_corner)
        right_eye = SingleEyeMovement(right_eye_upper_lid, right_eye_lower_lid, right_eye_left_corner,
                                      right_eye_right_corner)
        return cls(left_eye, right_eye)

    @classmethod
    def _landmark_to_vector(cls, landmark):
        return np.array([landmark.x, landmark.y, landmark.z])


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


if __name__ == "__main__":
    face_mesh = FaceMesh()
    visualization = Visualization()
    attention = Attention(
        RegionBoundary(30, 30, -8, -30),  # green region
        RegionBoundary(35, 50, -10, -50))  # yellow region
    sleepiness = Sleepiness(0.3, 0.05)
    talk_checker = TalkChecker()
    storage = Storage(10)

    attention_center = AttentionCenter(attention.yellow_region_boundary, EMA_alpha=0.005)
    MA_detected_region = None
    stationary_detected_region = None

    start_time = time.time()

    data_to_save = {"time_since_start": [],
                    "yaw": [],
                    "pitch": [],
                    "MA_prediction": [],
                    "stationary_prediction": []}

    while True:

        success, image = visualization.read_image()
        if not success:
            break

        landmarks = face_mesh.get_face_landmarks(image)
        if landmarks is None:
            head_position = None
            lips_movement = None
            eyes_movement = None
            iris_coordinates = None
        else:
            head_position = HeadPosition.from_landmarks_list(landmarks)
            lips_movement = LipsMovement.from_landmarks_list(landmarks)
            eyes_movement = EyesMovement.from_landmarks_list(landmarks)
            attention_center.UpdateAttention_EMA(MA_detected_region)
            iris_coordinates = get_iris_coordinates(image)

        attention.update_head_position(head_position)
        attention.update_attention_center(attention_center)
        sleepiness.update(eyes_movement)
        talk_checker.update(lips_movement)

        MA_detected_region = attention.get_detected_region_from_saved_position_as_vector()
        storage.add(MA_detected_region)
        stationary_detected_region = attention.get_detected_region_from_saved_position()

        data_to_save["time_since_start"].append(time.time() - start_time)
        if head_position:
            data_to_save["yaw"].append(head_position.yaw)
            data_to_save["pitch"].append(head_position.pitch)
        else:
            data_to_save["yaw"].append(np.nan)
            data_to_save["pitch"].append(np.nan)

        data_to_save["MA_prediction"].append(MA_detected_region.name)
        data_to_save["stationary_prediction"].append(stationary_detected_region.name)

        visualization.show(head_position=head_position, region=MA_detected_region, attention_center=attention_center,
                           talk_checker=talk_checker, sleepiness=sleepiness, storage=storage,
                           iris_coordinates=iris_coordinates)
        if visualization.is_return_key_pressed():
            if not path.exists("./data"):
                mkdir("data")
            dataframe_to_save = pd.DataFrame.from_dict(data_to_save)
            dataframe_to_save.to_csv(f"./data/{int(time.time())}.csv", index=False)
            break
