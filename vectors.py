import mediapipe as mp
import numpy as np
import cv2
from math import atan2, pi


def normalize(v):
    return v / np.sqrt(np.sum(v**2))


def simple_attention(dist):
    if dist < 0.15:
        return 3
    elif dist < 0.4:
        return 2
    else:
        return 1


mp_face_mesh = mp.solutions.face_mesh

vc = cv2.VideoCapture(0)
theta_lr = 0
theta_du = 0
theta_lu = 0

theta_x = 0
theta_y = 0
theta_z = 0


alpha = 0.01


center = np.array([0, 0, -1])
normal = center.copy()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


attention_level = 0
dist = 0

EMA_center = center.copy()
EMA_alpha = 0.01


while True:
    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        success, image = vc.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image)
        try:
            landmarks = results.multi_face_landmarks[0].landmark
            left = landmarks[234]
            right = landmarks[454]
            down = landmarks[151]
            up = landmarks[9]

            lr_vec = np.array([right.x - left.x, right.z - left.z])
            lu_vec = np.array([right.x - left.x, right.y - left.y])
            du_vec = np.array([up.y - down.y, up.z - down.z])

            horizontal = np.array([right.x - left.x, right.y - left.y, right.z - left.z])
            vertical = np.array([up.x - down.x, up.y - down.y, up.z - down.z])

            normal = np.cross(vertical, horizontal)

            normal = normalize(normal)
            """
            simple_center = alpha * normal_normalized + center
            simple_center = normalize(simple_center)
            """
            EMA_center = normalize(normal * EMA_alpha + (1-EMA_alpha) * EMA_center)

            theta_lr = atan2(lr_vec[1], lr_vec[0]) * 180 / pi
            theta_du = atan2(du_vec[1], du_vec[0]) * 180 / pi * -1
            theta_lu = atan2(lu_vec[1], lu_vec[0]) * 180 / pi

            theta_x = - 90 + atan2(normal[2], normal[1]) * 180 / pi * -1
            theta_y = 90 + atan2(normal[2], normal[0]) * 180 / pi
            theta_z = atan2(normal[1], normal[0]) * 180 / pi

            dist = np.linalg.norm(normal - EMA_center)

            attention_level = simple_attention(dist)

        except:
            cv2.putText(image, 'LOST AT: ', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            attention_level = 0

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(image, f"YAW: {theta_lr:.0f}, {theta_y:.0f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, f"ROLL: {theta_lu:,.0f}, {theta_z:.0f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"PITCH: {theta_du:.0f}, {theta_x:.0f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        """ cv2.putText(image, f"horizontal: {horizontal[0]:.2f}, {horizontal[1]:.2f}, {horizontal[2]:.2f}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"vertical: {vertical[0]:.2f}, {vertical[1]:.2f}, {vertical[2]:.2f}", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)"""

        cv2.putText(image, f"normal: {normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"center: {EMA_center[0]:.2f}, {EMA_center[1]:.2f}, {EMA_center[2]:.2f}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(image, f"dist: {dist:.4f}", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"attention: {attention_level}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # mp_drawing.draw_landmarks(image, up, mp_holistic.FACE_CONNECTIONS)
        """
        cv2.putText(image, f"UP: {up.x:.2f}, {up.y:.2f}, {up.z:.2f}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"DOWN: {down.x:.2f}, {down.y:.2f}, {down.z:.2f}", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"RIGHT: {right.x:.2f}, {right.y:.2f}, {right.z:.2f}", (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f"LEFT: {left.x:.2f}, {left.y:.2f}, {left.z:.2f}", (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        """

        # print(count)

        cv2.imshow('vid', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
