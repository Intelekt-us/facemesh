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
            self.detected_region = 0
        elif not self.head_position.is_in_region(self.yellow_region_boundary):
            self.detected_region = 1
        else:
            if self.head_position.is_in_region_as_vector(self.green_region_boundary, attention_center.vector):
                self.detected_region = 3
            else:
                self.detected_region = 2
        return self.detected_region

    def get_detected_region_from_saved_position(self):
        self.detected_region = None
        if not self.head_position:
            self.detected_region = 0
        else:
            if self.head_position.is_in_region(self.green_region_boundary):
                self.detected_region = 3
            elif self.head_position.is_in_region(self.yellow_region_boundary):
                self.detected_region = 2
            else:
                self.detected_region = 1
        return self.detected_region


def get_attention(image):
    attention = Attention(
        RegionBoundary(30, 30, -8, -30), #green region
        RegionBoundary(35, 50, -10, -50)) #yellow region
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    reverted_colors_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mesh_results = face_mesh.process(reverted_colors_image)
    try:
        landmarks = mesh_results.multi_face_landmarks[0].landmark
    except:
        landmarks = None

    if landmarks is None:
        head_position = None
    else:
        head_position = HeadPosition.from_landmarks_list(landmarks)

    attention.update_head_position(head_position)
    attention_result = attention.get_detected_region_from_saved_position()
    return attention_result