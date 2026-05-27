import time

import cv2
import PIL.Image as Image
from PIL import ImageDraw, ImageFont
import numpy as np


class UI:
    def __init__(self):
        self.frame = None
        self.prediction = None
        self.gestures = {
            0: "Здравей",
            1: "Казвам се",
            2: "Работя"
        }
    def updateFrame(self, frame):
        self.frame = frame
    def updatePrediction(self, prediction):
        self.prediction = prediction

    def drawUI(self):

        if self.frame is None:
            return

        frame = self.frame.copy()

        #top bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)



        #prediction
        if self.prediction is not None:
            text = f"Prediction: {self.gestures.get(self.prediction.item())}"

            img_pil = Image.fromarray(frame)

            draw = ImageDraw.Draw(img_pil)

            font = ImageFont.truetype("arial.ttf", 32)

            draw.text((20, 20), text, font=font)

            frame = np.array(img_pil)

        cv2.imshow("AI Vision", frame)

    def countdown(self, seconds=3):
        for i in range(seconds, 0, -1):
            print(i)
            time.sleep(1)

        print("Go!")