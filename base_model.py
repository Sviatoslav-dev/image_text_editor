import math
import os

import cv2
import keras_ocr
import numpy as np
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
        if resized_image.shape[1] > 50:
            resized_image = resized_image[:, :50]
        prediction = self.fonts_model.predict(np.array([resized_image]))[0]
        font_num = np.argmax(prediction)
        return fonts[int(font_num)]

    def get_mean_color(self, image):
        mask = self._segmentate_text(image)
        mean = cv2.mean(image, mask)
        return int(mean[0]), int(mean[1]), int(mean[2])
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

    def clear_text(self, image, prediction):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        pts = np.array([[int(point[0]), int(point[1])] for point in prediction], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
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
        draw.text((int(x) + 5, int(y) - 5), text, font=font, fill=color)
        img = np.array(img_pil)
        return img

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

        tolerance = 5

        for text, box in predictions:
            base_y = (box[0][1] + box[2][1]) / 2
            if abs(base_y - current_base_y) > tolerance:
                current_line = sorted(current_line, key=lambda x: x[1])
                lines.append(" ".join([word[0] for word in current_line]))
                current_line = []
                current_base_y = base_y
            current_line.append((text, box[0][0]))

        if current_line:
            current_line = sorted(current_line, key=lambda x: x[1])
            lines.append(" ".join([word[0] for word in current_line]))

        return lines
