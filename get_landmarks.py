
import mediapipe as mp
import numpy as np
import cv2
from math import acos, pi
mp_face_mesh = mp.solutions.face_mesh

vc = cv2.VideoCapture(0)
while True:
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:


        success, image = vc.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        blank_image = np.zeros((500,500,3), np.uint8)
        left = results.multi_face_landmarks[0].landmark[234]
        #[234]
        right = results.multi_face_landmarks[0].landmark[454]
        #[454]
        lr_vec = np.array([left.x - right.x, left.z - right.z])
        reference_vec = np.array([-1,0])

        theta = acos(np.dot(lr_vec, reference_vec) / np.linalg.norm(lr_vec)) * 180 / pi
        cv2.putText(image,f"{theta:.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
        cv2.imshow('vid',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   


# create 

