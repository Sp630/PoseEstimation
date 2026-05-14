import torch
import Training
from collections import deque
import cv2
import mediapipe as mp
import mediapipe.tasks.python.vision.drawing_utils
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from Drawing import draw_Landmarks, draw_connections
import time
import numpy as np
import UI
from Scripts.UI import UI
from DataCollection import landmarkNormalization

model = Training.Model()
model.load_state_dict(torch.load("../TrainedModels/model5.pth"))
model.eval()

ui = UI()

detection_result_hands = None
detection_result_pose = None

shoulder_pos_global = None
shoulder_width_global = None

def callback_hands(result, output_image, timestamp_ms):
    global detection_result_hands
    detection_result_hands = result

def callback_pose(result, output_image, timestamp_ms):
    global detection_result_pose, shoulder_pos_global, shoulder_width_global
    detection_result_pose = result
    sholder_left = detection_result_pose.pose_landmarks[0][11]
    sholder_right = detection_result_pose.pose_landmarks[0][12]
    shoulder_pos_global = [sholder_right.x - sholder_left.x, sholder_right.y - sholder_left.y, sholder_right.z - sholder_left.z]
    shoulder_width_global = np.sqrt((sholder_left.x - sholder_right.x) ** 2 + (sholder_left.y - sholder_right.y) ** 2)

hand_counter = 0
pose_counter = 0
feature =  np.zeros([60, 225], dtype= np.float32)
frame = np.zeros((225), dtype= np.float32)
frame_completion = 0
sequence = deque(maxlen=60)
dataset = []
def add_to_frame(landmarks, type):
    global frame_completion
    if type == 'pose':
        frame[0:99] = landmarks
        frame_completion += 1

    elif type == 'hand':
        frame[99:255] = landmarks
        frame_completion += 1
    if frame_completion == 2:
        sequence.append(frame)
        frame_completion = 0


def add_previous_frame(type):
    global frame_completion
    if not sequence:
        return
    if type == 'pose':
        previous = sequence[-1][0:99]
        frame[0:99] = previous
        frame_completion += 1
    elif type == 'hand':
        previous = sequence[-1][99:255]
        frame[99:255] = previous
        frame_completion += 1
    if frame_completion == 2:
        sequence.append(frame)
        frame_completion = 0

def predict_irl():
    if len(sequence) >= 60:
        X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = model(X)
            pred = torch.argmax(out, dim=1)
            #print(pred)
            ui.updatePrediction(pred)

base_options_hands = python.BaseOptions(model_asset_path ="../Models/hand_landmarker.task")
mode = python.vision.RunningMode
options_hands = vision.HandLandmarkerOptions(base_options=base_options_hands, running_mode= mode.LIVE_STREAM, num_hands=2, result_callback= callback_hands)
hands_detector = vision.HandLandmarker.create_from_options(options_hands)

base_options_pose = python.BaseOptions(model_asset_path ="../Models/pose_landmarker_heavy.task")
mode = python.vision.RunningMode
options_pose = vision.PoseLandmarkerOptions(base_options= base_options_pose, running_mode= mode.LIVE_STREAM, num_poses=1, result_callback= callback_pose)
pose_detector = vision.PoseLandmarker.create_from_options(options_pose)

start_time = time.time()
cap = cv2.VideoCapture(0)
record = True
while record:
    success, img = cap.read()
    if success:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        time_ms = int((time.time() - start_time) * 1000)
        hands_detector.detect_async(img_mp, time_ms)
        if(detection_result_hands is not None and len(detection_result_hands.hand_landmarks) > 0 and shoulder_width_global is not None):
            hand = 1
            for lm in detection_result_hands.hand_landmarks:
                draw_img =  draw_Landmarks(img, lm)

            arr = np.zeros([2, 21, 3])
            for i, hand in enumerate(detection_result_hands.hand_landmarks):
                if detection_result_hands.handedness[i][0].category_name == "Left":
                    arr[0] = [landmarkNormalization(lm, shoulder_pos_global, shoulder_width_global) for lm in hand]
                else:
                    arr[1] = [landmarkNormalization(lm, shoulder_pos_global, shoulder_width_global) for lm in hand]

            flat = arr.reshape(-1)
            add_to_frame(flat, 'hand')


        elif shoulder_width_global is not None:
            add_previous_frame('hand')


        pose_detector.detect_async(img_mp, time_ms)
        if(detection_result_pose is not None and len(detection_result_pose.pose_landmarks) > 0):
            for lm in detection_result_pose.pose_landmarks:
                draw_img = draw_Landmarks(img, lm)
                draw_img = draw_connections(img, lm)
            arr = np.array([[landmarkNormalization(lm, shoulder_pos_global, shoulder_width_global) for lm in pose] for pose in detection_result_pose.pose_landmarks])
            flat = arr.reshape(-1)
            if shoulder_width_global is not None:
                add_to_frame(flat, 'pose')
        elif shoulder_width_global is not None:
            add_previous_frame('pose')

        predict_irl()



        img = cv2.flip(img, 1)
        #cv2.imshow("Image", img)
        ui.updateFrame(img)
        ui.drawUI()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #pose_landmarker.close()
            np.save(f"../Dataset/Казвам се/s10.npy", feature)

            break