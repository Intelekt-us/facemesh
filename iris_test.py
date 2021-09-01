import cv2
import numpy as np
import dlib
from custom.iris_lm_depth import from_landmarks_to_iris
import mediapipe as mp


points_idx = [33, 133, 362, 263, 61, 291, 199]
points_idx = list(set(points_idx))
points_idx.sort()

left_eye_landmarks_id = np.array([33, 133])
right_eye_landmarks_id = np.array([362, 263])

dist_coeff = np.zeros((4, 1))

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
SMALL_CIRCLE_SIZE = 1
LARGE_CIRCLE_SIZE = 2

#frame_height = frame.shape[0]
#frame_width = frame.shape[1]
#image_size = (frame_width, frame_height)
#focal_length = frame_width
#facemesh = mp.solutions.face_mesh.FaceMesh()


'''
vid = cv2.VideoCapture(0)
ret, frame = vid.read()
while True:
    ret, frame = vid.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_results = facemesh.process(frame_rgb)
    try:
        face_landmarks = mesh_results.multi_face_landmarks[0]
    except:
        continue


    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
    landmarks = landmarks.T




    left_eye_landmarks = from_landmarks_to_iris(
                    frame_rgb,
                    landmarks[:, left_eye_landmarks_id],
                    image_size,
                    is_right_eye=False,
                    focal_length=focal_length,
                )
    right_eye_landmarks = from_landmarks_to_iris(
                    frame_rgb,
                    landmarks[:, right_eye_landmarks_id],
                    image_size,
                    is_right_eye=True,
                    focal_length=focal_length,
                )



    size = 500
    blank_image = np.zeros((size, size, 3), np.uint8)
    landmarks = landmarks.T
    for point in landmarks:
        cv2.circle(frame, (int(point[0] * frame_width), int(point[1] * frame_height)), 0, (0,255,0), 2)
    for point in right_eye_landmarks:
        cv2.circle(frame, (int(point[0] * frame_width), int(point[1] * frame_height)), 0, (0,0,255), 3)
    for point in left_eye_landmarks:
        cv2.circle(frame, (int(point[0] * frame_width), int(point[1] * frame_height)), 0, (0,0,255), 3)
    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
'''


def get_iris_coordinates(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh() as facemesh:
        mesh_results = facemesh.process(frame_rgb)
        try:
            face_landmarks = mesh_results.multi_face_landmarks[0]
        except:
            return None
    
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    image_size = (frame_width, frame_height)
    focal_length = frame_width



    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
    landmarks = landmarks.T




    left_eye_landmarks = from_landmarks_to_iris(
                    frame_rgb,
                    landmarks[:, left_eye_landmarks_id],
                    image_size,
                    is_right_eye=False,
                    focal_length=focal_length,
                )
    right_eye_landmarks = from_landmarks_to_iris(
                    frame_rgb,
                    landmarks[:, right_eye_landmarks_id],
                    image_size,
                    is_right_eye=True,
                    focal_length=focal_length,
                )
    return (left_eye_landmarks, right_eye_landmarks)