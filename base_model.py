import math
import os
import random

import cv2
import keras_ocr
import numpy as np
from googletrans import Translator
from keras.models import load_model

from PIL import ImageFont, Image, ImageDraw

from font_predictions.fonts import fonts


class BaseImageModel:
    def __init__(self):
        self.pipeline = keras_ocr.pipeline.Pipeline()

        BASEDIR = "."
        MODEL_DIR = os.path.join(BASEDIR, "font_predictions/saved_models")

        model_name = os.path.join(MODEL_DIR, "font.model.02.keras")
        self.fonts_model = load_model(model_name)
        self.translator = Translator()

    def _polygon_to_box(self, polygon):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        min_x = min(min_x, min([coordinate[0] for coordinate in polygon]))
        min_y = min(min_y, min([coordinate[1] for coordinate in polygon]))
        max_x = max(max_x, max([coordinate[0] for coordinate in polygon]))
        max_y = max(max_y, max([coordinate[1] for coordinate in polygon]))
        return min_x, min_y, max_x - min_x, max_y - min_y

    def _unite_predictions(self, prediction_groups):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0

        for prediction in prediction_groups[0]:
            box = prediction[1]
            min_x = min(min_x, min([coordinate[0] for coordinate in box]))
            min_y = min(min_y, min([coordinate[1] for coordinate in box]))
            max_x = max(max_x, max([coordinate[0] for coordinate in box]))
            max_y = max(max_y, max([coordinate[1] for coordinate in box]))

        overall_box = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        return np.array(overall_box)

    def find_polygon_centroid(self, points):
        x_sum = 0
        y_sum = 0
        n = len(points)

        for x, y in points:
            x_sum += x
            y_sum += y

        centroid_x = x_sum / n
        centroid_y = y_sum / n

        return centroid_x, centroid_y

    def find_text(self, image):
        predictions = self.pipeline.recognize([image])
        return predictions, self._unite_predictions(predictions)

    def calculate_box_height(self, box):
        points = np.array(box)

        height1 = np.linalg.norm(points[0] - points[2])
        height2 = np.linalg.norm(points[1] - points[3])

        height = max(height1, height2)
        return height

    def predict_font(self, img: np.ndarray):
        height, weight, _ = img.shape
        k = height / 50
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_weight = int(weight / k)
        resized_image = cv2.resize(gray_image, (new_weight, 50))
        resized_image = np.reshape(resized_image, (50, new_weight, 1))
        if resized_image.shape[1] < 50:
            k = resized_image.shape[1] / 50
            new_width = int(resized_image.shape[0] / k)
            resized_image = cv2.resize(resized_image, (int(50 / k), new_width))
            resized_image = np.reshape(resized_image, (int(50 / k), new_width, 1))
            resized_image = resized_image[:50, :]
        size_x = resized_image.shape[1]
        if size_x > 50:
            start = random.randint(0, size_x - 50)
            resized_image = resized_image[:, start:start + 50]
        prediction = self.fonts_model.predict(np.array([resized_image]))[0]
        font_num = np.argmax(prediction)
        return fonts[int(font_num)]

    def find_dominant_color(self, image, mask=None, k=3):
        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)

        pixels = np.float32(image.reshape(-1, 3))

        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        return tuple(dominant)

    def get_mean_color(self, image):
        # mask = self._segmentate_text(image)
        # mean = cv2.mean(image, mask)
        # return int(mean[0]), int(mean[1]), int(mean[2])
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # mean_brightness = np.mean(gray_image)
        #
        # # _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # text_is_dark = mean_brightness > 128
        #
        # if mean_brightness > 128:
        #     _, binary_image = cv2.threshold(gray_image, 0, 255,
        #                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # else:
        #     _, binary_image = cv2.threshold(gray_image, 0, 255,
        #                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #
        # cv2.imshow("rrr", binary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # masked_text_region = cv2.bitwise_and(image, image, mask=binary_image)
        # mask_value = 0 if text_is_dark else 255
        # pixels = masked_text_region[binary_image == mask_value]
        # average_color = tuple(np.mean(pixels, axis=0)) if len(pixels) > 0 else (0, 0, 0)
        # return int(average_color[0]), int(average_color[1]), int(average_color[2])
        gray_region = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray_region], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        q = np.cumsum(hist_norm)

        optimal_threshold = np.argmax((q * (1 - q) * (np.arange(256) - q * np.arange(256)) ** 2))

        thresh_type = cv2.THRESH_BINARY_INV if optimal_threshold < 128 else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray_region, optimal_threshold, 255, thresh_type | cv2.THRESH_OTSU)

        mask_inv = cv2.bitwise_not(mask)
        text_pixels = cv2.bitwise_and(image, image, mask=mask_inv)
        mean_color = cv2.mean(text_pixels, mask=mask_inv)[:3]
        # cv2.imshow('Text Region', text_pixels)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # mean_color = self.find_dominant_color(image, mask=mask_inv)
        return int(mean_color[0]), int(mean_color[1]), int(mean_color[2])

    def clear_text(self, image, prediction):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        pts = np.array([[int(point[0]), int(point[1])] for point in prediction], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        image = cv2.inpaint(image, mask, 11, cv2.INPAINT_TELEA)
        return image

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

        ascent, descent = font.getmetrics()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        # text_height = text_bbox[3] - text_bbox[1]
        text_height = ascent - descent
        text_height *= 1.5

        start_x = x - text_width // 2
        start_y = y - text_height // 2
        draw.text((start_x, start_y), text, font=font, fill=color)
        img = np.array(img_pil)
        return np.array(img)

    def get_box_height(self, box):
        return int((box[1][3][1] - box[1][0][1]))  # * 0.8)

    def _segmentate_text(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 3, 2)

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

    def group_text_by_lines(self, predictions):
        predictions = sorted(predictions, key=lambda x: (x[1][0][1] + x[1][2][1]) / 2)

        lines = []
        current_line = []
        current_base_y = (predictions[0][1][0][1] + predictions[0][1][2][1]) / 2

        tolerance = 20

        for text, box in predictions:
            base_y = (box[0][1] + box[2][1]) / 2
            if abs(base_y - current_base_y) > tolerance:
                current_line = sorted(current_line, key=lambda x: x[1])
                print([word[0] for word in current_line])
                lines.append(" ".join([word[0] for word in current_line]))
                current_line = []
                current_base_y = base_y
            current_line.append((text, box[0][0]))

        if current_line:
            current_line = sorted(current_line, key=lambda x: x[1])
            lines.append(" ".join([word[0] for word in current_line]))

        return lines
