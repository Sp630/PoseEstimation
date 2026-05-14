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
shoulder_pos_global = None
shoulder_width_global = None

def callback_hands(result, output_image, timestamp_ms):
    global detection_result_hands, last_hand_landmark
    detection_result_hands = result
    if len(result.hand_landmarks) > 0:
        for i, hand in enumerate(result.hand_landmarks):
            arr = np.zeros([126], dtype=np.float32)
            if result.handedness[i][0].category_name == "Left":
                arr[0:63] = [coordinate for landmark in hand for coordinate in landmarkNormalization(landmark, shoulder_pos_global, shoulder_width_global)]
            else:
                arr[63:126] = [coordinate for landmark in hand for coordinate in landmarkNormalization(landmark, shoulder_pos_global, shoulder_width_global)]
        last_hand_landmark = arr

def callback_pose(result, output_image, timestamp_ms):
    global detection_result_pose, shoulder_pos_global, shoulder_width_global, last_pose_landmark
    detection_result_pose = result
    if len(result.pose_landmarks) > 0:
        last_pose_landmark = [coordinate for hand in result.pose_landmarks for landmark in hand for coordinate in landmarkNormalization(landmark, shoulder_pos_global, shoulder_width_global)]
    sholder_left = detection_result_pose.pose_landmarks[0][11]
    sholder_right = detection_result_pose.pose_landmarks[0][12]
    shoulder_pos_global = [sholder_right.x - sholder_left.x, sholder_right.y - sholder_left.y, sholder_right.z - sholder_left.z]
    shoulder_width_global = np.sqrt((sholder_left.x - sholder_right.x) ** 2 + (sholder_left.y - sholder_right.y) ** 2)
    print(f"Sholder position: {shoulder_pos_global}")


def landmarkNormalization(landmarks, shoulder_position, shoulder_width):
    if shoulder_position is None:
        print("WARNING, MISSING NORMALIZATION DATA; CANNOT NORMALIZE!!!")
        return [landmarks.x, landmarks.y, landmarks.z]
    x = (landmarks.x - shoulder_position[0]) / shoulder_width
    y = (landmarks.y - shoulder_position[1]) / shoulder_width
    z = (landmarks.z - shoulder_position[2]) / shoulder_width
    return [x, y, z]


hand_counter = 0
pose_counter = 0
feature =  np.zeros([60, 225], dtype= np.float32)
feature_list = []
dataset = []
last_time = 0
interval = 1.0 / 15
last_pose_landmark = None
last_hand_landmark = None

def save_to_file(last_pose_landmark, last_hand_landmark):
    frame = np.zeros([225], dtype= np.float32)
    frame[0:99] = last_pose_landmark if last_hand_landmark is not None else np.zeros([99], dtype= np.float32)
    frame[99:225] =  last_hand_landmark if last_hand_landmark is not None else np.zeros([126], dtype= np.float32)
    feature_list.append(frame)
    feature_array = np.array(feature_list)
    print(feature_array.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
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
    start_rec_time = time.time()
    while record:
        success, img = cap.read()
        if success:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

            time_ms = int((time.time() - start_time) * 1000)
            hands_detector.detect_async(img_mp, time_ms)
            if(last_hand_landmark is not None and shoulder_width_global is not None and time.time() - last_time > interval):
                #last_time = time.time()

                for lm in detection_result_hands.hand_landmarks:
                    draw_img =  draw_Landmarks(img, lm)

                save_to_file(last_pose_landmark, last_hand_landmark)


            pose_detector.detect_async(img_mp, time_ms)
            if(last_pose_landmark is not None and time.time() - last_time > interval):
                last_time = time.time()
                for lm in detection_result_pose.pose_landmarks:
                    draw_img = draw_Landmarks(img, lm)
                    draw_img = draw_connections(img, lm)
                if shoulder_width_global is not None:
                    save_to_file(last_pose_landmark, last_hand_landmark)


        img = cv2.flip(img, 1)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_rec_time > 3:
            np.save(f"../Dataset1/Казвам се/s30.npy", feature)
            break