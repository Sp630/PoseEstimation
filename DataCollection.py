import cv2
import mediapipe as mp
import mediapipe.tasks.python.vision.drawing_utils
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from Drawing import draw_Landmarks, draw_connections
import time

detection_result_hands = None
detection_result_pose = None

def callback_hands(result, output_image, timestamp_ms):
    print(f"Hands detected: {len(result.hand_landmarks)}")
    global detection_result_hands
    detection_result_hands = result

def callback_pose(result, output_image, timestamp_ms):
    global detection_result_pose
    detection_result_pose = result

hand_counter = 0
pose_counter = 0
feature =  np.zeros([60, 225], dtype= np.float32)
dataset = []
def save_to_file(landmarks, type):
    global hand_counter, pose_counter
    if type == 'pose':
        pose_counter += 1
        feature[pose_counter][0:99] = landmarks
    elif type == 'hand':
        hand_counter += 1
        feature[hand_counter][99:255] = landmarks

def save_previous(type):
    global pose_counter, hand_counter
    if type == 'pose':
        previous = feature[-1][0:99]
        pose_counter += 1
        feature[hand_counter][0:99] = previous
    elif type == 'hand':
        previous = feature[hand_counter][99:255]
        hand_counter += 1
        feature[hand_counter][99:255] = previous




base_options_hands = python.BaseOptions(model_asset_path = "Models/hand_landmarker.task")
mode = python.vision.RunningMode
options_hands = vision.HandLandmarkerOptions(base_options=base_options_hands, running_mode= mode.LIVE_STREAM, num_hands=2, result_callback= callback_hands)
hands_detector = vision.HandLandmarker.create_from_options(options_hands)

base_options_pose = python.BaseOptions(model_asset_path = "Models/pose_landmarker_heavy.task")
mode = python.vision.RunningMode
options_pose = vision.PoseLandmarkerOptions(base_options= base_options_pose, running_mode= mode.LIVE_STREAM, num_poses=1, result_callback= callback_pose)
pose_detector = vision.PoseLandmarker.create_from_options(options_pose)

start_time = time.time()
cap = cv2.VideoCapture(0)
record = True
while record:
    print(hand_counter)
    success, img = cap.read()
    if success:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        time_ms = int((time.time() - start_time) * 1000)
        hands_detector.detect_async(img_mp, time_ms)
        if(detection_result_hands is not None and len(detection_result_hands.hand_landmarks) > 0):
            hand = 1
            for lm in detection_result_hands.hand_landmarks:
                draw_img =  draw_Landmarks(img, lm)

            arr = np.zeros([2, 21, 3])
            for i, hand in enumerate(detection_result_hands.hand_landmarks):
                arr[i] = [[lm.x, lm.y, lm.z] for lm in hand]
            flat = arr.reshape(-1)
            save_to_file(flat, 'hand')


        else:
            save_previous('hand')


        pose_detector.detect_async(img_mp, time_ms)
        if(detection_result_pose is not None and len(detection_result_pose.pose_landmarks) > 0):
            for lm in detection_result_pose.pose_landmarks:
                draw_img = draw_Landmarks(img, lm)
                draw_img = draw_connections(img, lm)
            arr = np.array([[[lm.x, lm.y, lm.z] for lm in pose] for pose in detection_result_pose.pose_landmarks])
            flat = arr.reshape(-1)
            save_to_file(flat, 'pose')
        else:
            save_previous('pose')

    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or hand_counter == 59:
        break