
import mediapipe as mp
import numpy as np
import cv2
from math import atan2, acos, pi
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
        left = results.multi_face_landmarks[0].landmark[234]
        right = results.multi_face_landmarks[0].landmark[454]
        down = results.multi_face_landmarks[0].landmark[151]
        up = results.multi_face_landmarks[0].landmark[9]
        lr_vec = np.array([right.x - left.x, right.z - left.z])
        lu_vec = np.array([right.x - left.x, right.y - left.y])
        du_vec = np.array([up.y - down.y, up.z - down.z])

        theta_lr = atan2(lr_vec[1], lr_vec[0]) * 180 / pi
        theta_lu = atan2(lu_vec[1], lu_vec[0]) * 180 / pi
        theta_du = atan2(du_vec[1], du_vec[0]) * 180 / pi
        cv2.putText(image,f"YAW: {theta_lr:.0f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
        cv2.putText(image,f"ROLL: {theta_lu:.0f}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.putText(image,f"PITCH: {theta_du:.0f}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
        cv2.imshow('vid',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   


# create 

