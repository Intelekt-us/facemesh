import mediapipe as mp
import numpy as np
import cv2
from math import atan2, acos, pi
from enum import Enum
from dataclasses import dataclass
import time
import pandas as pd
from os import path, mkdir


class HeadPosition:
    def __init__(self, direction):
        self.direction = direction
        self.yaw = self._yaw(direction)
        self.pitch = self._pitch(direction)

    def is_in_region(self, region_boundary):
        return self.pitch < region_boundary.up and self.pitch > region_boundary.down \
            and self.yaw > region_boundary.left and self.yaw < region_boundary.right

    def is_in_region_as_vector(self, region_boundary, attention_center_vector):
        # return np.linalg.norm(self.direction - attention_center_vector) < 0.3
        return self.pitch - self._pitch(attention_center_vector) < region_boundary.up and self.pitch - self._pitch(attention_center_vector) > region_boundary.down \
            and self.yaw - self._yaw(attention_center_vector) > region_boundary.left and self.yaw - self._yaw(attention_center_vector) < region_boundary.right
    
    @classmethod
    def _yaw(cls, direction):
        return 90 + atan2(direction[2], direction[0]) * 180 / pi

    @classmethod
    def _pitch(cls, direction):
        return -90 + atan2(direction[2], direction[1]) * 180 / pi * -1

    
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
        
        return cls(normal)

    @classmethod
    def _normalize(cls, vector):
        return vector / np.sqrt(np.sum(vector**2))
    


class Visualization:
    def __init__(self, show_position=True, show_fps=True, show_region=True, show_attention_center = True):
        self.vc = cv2.VideoCapture(0)
        self.show_position = show_position
        self.show_attention_center = show_attention_center
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
        cv2.putText(self.image,f"PITCH: {head_position.pitch:.1f}", (5,60), self.default_font, 0.8, (255,0,0), 2)

    def display_direction(self, head_position):
        cv2.putText(self.image,f"[{head_position.direction[0]:.2f}, {head_position.direction[1]:.2f}, {head_position.direction[2]:.2f}] (HEAD DIRECTION)", (5,420), self.default_font, 0.8, (0,0,255), 2)
   
    def display_attention_center(self, attention_center):
        cv2.putText(self.image,f"[{attention_center.vector[0]:.2f}, {attention_center.vector[1]:.2f}, {attention_center.vector[2]:.2f}] (ATTENTION VECTOR)", (5,450), self.default_font, 0.8, (0,0,255), 2)
    
    def display_fps(self):
        cv2.putText(self.image, f"FPS: {self.fps}", (10, 120), self.default_font, 1, (0,255,0), 2)

    def display_detected_region(self, region):
        cv2.putText(self.image, f'REGION: {region.name}', (10, 150), self.default_font, 1, region.value, 2)

    def show(self, head_position=None, region=None, attention_center=None):
        if self.show_position and head_position:
            self.display_position(head_position)
            self.display_direction(head_position)
        if self.show_attention_center and attention_center:
            self.display_attention_center(attention_center)
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


class AttentionCenter:
    def __init__(self, yellow_region_boundary, EMA_alpha = 0.01):
        self.vector = np.array([0, 0, -1])
        self.EMA_alpha = EMA_alpha
        self.yellow_region_boundary = yellow_region_boundary

    def UpdateAttention_dummy(self):
        self.vector = np.array([0, 0, -1])

    def UpdateAttention_EMA(self, detected_region):
        if detected_region == Regions.YELLOW or detected_region == Regions.GREEN:
            self.vector = AttentionCenter._normalize(head_position.direction * self.EMA_alpha + (1-self.EMA_alpha) * self.vector)

    @classmethod
    def _normalize(cls, vector):
        return vector / np.sqrt(np.sum(vector**2))



class Attention:
    def __init__(self, green_region_boundary, yellow_region_boundary, attention_center=np.array([0,0,-1])):
        self.green_region_boundary = green_region_boundary # solution using angles
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



if __name__ == "__main__":
    face_mesh = FaceMesh()
    visualization = Visualization()
    attention = Attention(
        RegionBoundary(30, 30, -8, -30), #green region
        RegionBoundary(35, 50, -10, -50)) #yellow region

    attention_center = AttentionCenter(attention.yellow_region_boundary, EMA_alpha=0.005)
    detected_region = None

    start_time = time.time()

    data_to_save = {"time_since_start":[],
                    "yaw":[],
                    "pitch":[],
                    "prediction":[]}

    while True:

        success, image = visualization.read_image()
        if not success:
            break

        landmarks = face_mesh.get_face_landmarks(image)
        if landmarks is None:
            head_position = None
        else:
            head_position = HeadPosition.from_landmarks_list(landmarks)
            attention_center.UpdateAttention_EMA(detected_region)

        attention.update_head_position(head_position)
        attention.update_attention_center(attention_center)
        detected_region = attention.get_detected_region_from_saved_position_as_vector()
        

        data_to_save["time_since_start"].append(time.time() - start_time)
        if head_position:
            data_to_save["yaw"].append(head_position.yaw)
            data_to_save["pitch"].append(head_position.pitch)
        else:
            data_to_save["yaw"].append(np.nan)
            data_to_save["pitch"].append(np.nan)
            
        data_to_save["prediction"].append(detected_region.name)


        visualization.show(head_position=head_position, region=detected_region, attention_center=attention_center)
        if visualization.is_return_key_pressed():
            if not path.exists("./data"):
                mkdir("data")
            dataframe_to_save = pd.DataFrame.from_dict(data_to_save)
            dataframe_to_save.to_csv(f"./data/{int(time.time())}.csv", index=False)
            break