import math
import os

import cv2
import keras_ocr
import numpy as np
from keras.models import load_model

from font_from_image_main.fonts import fonts
from PIL import ImageFont, Image, ImageDraw


class BaseImageModel:
    def __init__(self):
        self.pipeline = keras_ocr.pipeline.Pipeline()

        BASEDIR = "."
        MODEL_DIR = os.path.join(BASEDIR, "font_from_image_main/saved_models")

        model_name = os.path.join(MODEL_DIR, "font.model.02.keras")
        self.fonts_model = load_model(model_name)

    def find_text(self, image):
        predictions = self.pipeline.recognize([image])
        return predictions

    def predict_font(self, img: np.ndarray):
        height, weight, _ = img.shape
        k = height / 50
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_weight = int(weight / k)
        resized_image = cv2.resize(gray_image, (new_weight, 50))
        resized_image = np.reshape(resized_image, (50, new_weight, 1))
        if resized_image.shape[1] > 50:
            resized_image = resized_image[:, :50]
        prediction = self.fonts_model.predict(np.array([resized_image]))[0]
        font_num = np.argmax(prediction)
        return fonts[int(font_num)]

    def get_mean_color(self, image):
        mask = self._segmentate_text(image)
        mean = cv2.mean(image, mask)
        return int(mean[0]), int(mean[1]), int(mean[2])

    def clear_text(self, image, prediction_group=None, box=None):
        box = prediction_group[0] if box is None else box
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = self._midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = self._midpoint(x0, y0, x3, y3)
        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)

        img_inpainted = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
        return img_inpainted

    def draw_text(self, image, text, x, y, h, color=(255, 255, 255), font="Arial"):
        # print("xy", x, y)
        # print("h = ", h)
        # thickness = 2
        # text_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, h, thickness)
        # return cv2.putText(
        #     image, text, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #     text_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        # )

        print(text, x, y, h, color, font)
        fontpath = f"data/fonts/{font}.ttf"
        font = ImageFont.truetype(fontpath, int(h * 1.3))
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text((int(x) + 5, int(y) - 5), text, font=font, fill=color)
        img = np.array(img_pil)
        return img

    def get_box_height(self, box):
        return int((box[1][3][1] - box[1][0][1]))  # * 0.8)

    def _segmentate_text(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        connected_text = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(connected_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_mask = np.zeros_like(adaptive_thresh)

        cv2.drawContours(text_mask, contours, -1, (255), thickness=cv2.FILLED)
        mask = text_mask - connected_text
        return mask

    def _midpoint(self, x1, y1, x2, y2):
        x_mid = int((x1 + x2) / 2)
        y_mid = int((y1 + y2) / 2)
        return x_mid, y_mid
