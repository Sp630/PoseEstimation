import cv2
import mediapipe as mp
import mediapipe.tasks.python.vision.drawing_utils
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_Landmarks(img, landmarks):
    h, w, _ = img.shape
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    return img

def draw_connections(img, landmarks):
    connections = python.vision.PoseLandmarksConnections.POSE_LANDMARKS
    h, w, _ = img.shape
    for conn in connections:
        start = conn.start
        end = conn.end
        x1 = int(landmarks[start].x * w)
        x2 = int(landmarks[end].x * w)
        y1 = int(landmarks[start].y * h)
        y2 = int(landmarks[end].y * h)

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img