import cv2
import mediapipe as mp
import mediapipe.tasks.python.vision.drawing_utils
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
    success, img = cap.read()
    if success:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        time_ms = int((time.time() - start_time) * 1000)
        hands_detector.detect_async(img_mp, time_ms)
        print(detection_result_hands)
        if(detection_result_hands is not None and len(detection_result_hands.hand_landmarks) > 0):
            print("IN IFFF")
            for lm in detection_result_hands.hand_landmarks:
                draw_img =  draw_Landmarks(img, lm)

        pose_detector.detect_async(img_mp, time_ms)
        if(detection_result_pose is not None and len(detection_result_pose.pose_landmarks) > 0):
            for lm in detection_result_pose.pose_landmarks:
                draw_img = draw_Landmarks(img, lm)
                draw_img = draw_connections(img, lm)

    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
