import mediapipe as mp
import numpy as np
import cv2
from math import atan2, acos, pi
from enum import Enum
from dataclasses import dataclass
import time

class HeadPosition:
    def __init__(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def is_in_region(self, region_boundary):
        return self.pitch < region_boundary.up and self.pitch > region_boundary.down \
            and self.yaw > region_boundary.left and self.yaw < region_boundary.right


    # automatic addition, substraction etc. of HeadPosition instances (allows for calculating average more easily)
    def __add__(self, p):
        if type(self) == type(self):
            return HeadPosition(self.roll + p.roll, self.pitch + p.pitch, self.yaw + p.yaw)
        else:
            return HeadPosition(self.roll + p, self.pitch + p, self.yaw + p)

    def __sub__(self, p):
        if type(self) == type(self):
            return HeadPosition(self.roll - p.roll, self.pitch - p.pitch, self.yaw - p.yaw)
        else:
            return HeadPosition(self.roll - p, self.pitch - p, self.yaw - p)

    def __mul__(self, p):
        if type(self) == type(self):
            return HeadPosition(self.roll * p.roll, self.pitch * p.pitch, self.yaw * p.yaw)
        else:
            return HeadPosition(self.roll * p, self.pitch * p, self.yaw * p)
    
    def __truediv__(self, p):
        if type(self) == type(self):
            return HeadPosition(self.roll / p.roll, self.pitch / p.pitch, self.yaw / p.yaw)
        else:
            return HeadPosition(self.roll / p, self.pitch / p, self.yaw / p)

    @classmethod
    def from_landmarks_list(cls, landmarks):
        left_ear = landmarks[234]
        right_ear = landmarks[454]
        chin = landmarks[151]
        forehead = landmarks[9]
        roll_vector = np.array([right_ear.x - left_ear.x, right_ear.y - left_ear.y])
        pitch_vector = np.array([forehead.y - chin.y, chin.z - forehead.z])
        yaw_vector = np.array([right_ear.x - left_ear.x, right_ear.z - left_ear.z])
        roll = cls._vector_to_angle(roll_vector)
        pitch = cls._vector_to_angle(pitch_vector)
        yaw = cls._vector_to_angle(yaw_vector)
        return cls(roll, pitch, yaw)
    
    @classmethod
    def _vector_to_angle(cls, vector):
        return atan2(vector[1], vector[0]) * 180 / pi


class Visualization:
    def __init__(self, show_position=True, show_fps=True, show_region=True):
        self.vc = cv2.VideoCapture(0)
        self.show_position = show_position
        self.show_fps = show_fps
        self.show_region = show_region
        self.last_time = time.time()
        self.fps = 0
        self.default_font = cv2.FONT_HERSHEY_SIMPLEX

    def read_image(self):
        success, self.image = self.vc.read()
        self._update_fps()
        return (success, self.image)

    def _update_fps(self):
        current_time = time.time()
        self.fps = int(1 / (current_time - self.last_time))
        self.last_time = current_time

    def display_position(self, head_position):      
        cv2.putText(self.image,f"YAW: {head_position.yaw:.1f}", (5,30), self.default_font, 0.8, (255,0,0), 2)
        cv2.putText(self.image,f"ROLL: {head_position.roll:.1f}", (5,60), self.default_font, 0.8, (255,0,0), 2)
        cv2.putText(self.image,f"PITCH: {head_position.pitch:.1f}", (5,90), self.default_font, 0.8, (255,0,0), 2)

    def display_fps(self):
        cv2.putText(self.image, f"FPS: {self.fps}", (10, 120), self.default_font, 1, (0,255,0), 2)

    def display_detected_region(self, region):
        cv2.putText(self.image, f'REGION: {region.name}', (10, 150), self.default_font, 1, region.value, 2)

    def show(self, head_position=None, region=None):
        if self.show_position and head_position:
            self.display_position(head_position)
        if self.show_fps:
            self.display_fps()
        if self.show_region and region:
            self.display_detected_region(region)
        cv2.imshow("vid", self.image)

    def is_return_key_pressed(self):
        return cv2.waitKey(1) & 0xFF == ord('q')


class FaceMesh:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence)

    def get_face_landmarks(self, image):
        reverted_colors_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(reverted_colors_image)
        try:
            return mesh_results.multi_face_landmarks[0].landmark
        except:
            return None

    
class Regions(Enum):
    GREEN = (0,255,0)
    YELLOW = (0,255,255)
    RED = (0,0,255)
    NOT_PRESENT = (0,0,0)


class RegionBoundary:
    def __init__(self, up, right, down, left):
        self.up = up
        self.down = down
        self.right = right
        self.left = left


class Attention:
    def __init__(self, green_region_boundary, yellow_region_boundary):
        self.green_region_boundary = green_region_boundary
        self.yellow_region_boundary = yellow_region_boundary
        self.head_position = None
        self.detected_region = None

    def set_green_region_boundary(self, boundary):
        self.green_region_boundary = boundary

    def update_head_position(self, new_position):
        self.head_position = new_position

    def get_detected_region_from_saved_position(self):
        self.detected_region = None
        if not self.head_position:
            self.detected_region = Region.NOT_SEEN
        else:
            if self.head_position.is_in_region(self.green_region_boundary):
                self.detected_region = Region.GREEN
            elif self.head_position.is_in_region(self.yellow_region_boundary):
                self.detected_region = Region.YELLOW
            else:
                self.detected_region = Region.RED
        return self.detected_region



if __name__ == "__main__":
    face_mesh = FaceMesh()
    visualization = Visualization()
    attention = Attention(
        RegionBoundary(30, 30, -8, -30), #green region
        RegionBoundary(35, 50, -10, -50)) #yellow region

    while True:

        success, image = visualization.read_image()
        if not success:
            break

        landmarks = face_mesh.get_face_landmarks(image)
        if landmarks is None:
            head_position = None
        else:
            head_position = HeadPosition.from_landmarks_list(landmarks)
        attention.update_head_position(head_position)
        detected_region = attention.get_detected_region_from_saved_position()

        visualization.show(head_position=head_position, region=detected_region)
        if visualization.is_return_key_pressed():
            break

