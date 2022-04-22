import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class handRaise:
    def __init__(self, min_visibility, mouth_y, right_hand_landmark, left_hand_landmark):
        self.mouth_y = mouth_y
        self.right_hand_landmark = right_hand_landmark
        self.left_hand_landmark = left_hand_landmark
        self.if_raise = False
        self.min_visibility = min_visibility
    def if_landmark_up(self, hand_landmark):
        if(hand_landmark.visibility > self.min_visibility and hand_landmark.y < self.mouth_y): return True
        else: return False
    def check_hand_raised(self):
        if(self.if_landmark_up(self.right_hand_landmark) or self.if_landmark_up(self.left_hand_landmark)):
            self.if_raise = True
            print("Hand is raised!")
        else:
            self.if_raise = False
            print("-")
        print(self.if_raise)
        return self.if_raise

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0) as pose:
    #model_complexity nominalnie zwieksza dokladnosc, ale tez wydaje mi sie ze spowalnia zmiane atrybutu visibility
    # warto sprawdzic dzialanie detection i tracking confidence na responsywnosc
  while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    hand_raise = handRaise(0.8, results.pose_landmarks.landmark[9].y, results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[15])
        # pytanie jakiego min visibility (pierwsza wartosc) i landmarka na głowie (pierwszy z landmarkow) użyć- kalibracja i user experience
    hand_raise.check_hand_raised()

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27: #ESC żeby zakończyć
        break
cap.release()