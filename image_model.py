import cv2
import numpy as np
import pyperclip
import pytesseract
from PIL import ImageFont, Image, ImageDraw

from base_editor.base_model import BaseImageModel


class ImageModel(BaseImageModel):
    def __init__(self):
        super().__init__()
        self.img = None
        self.translate_option = ("en", "uk")

    def read_image(self, path):
        self.img = cv2.imread(path, 1)
        self.start_image = np.copy(self.img)

    def replace_text(self, new_text, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        prediction_groups, united_groups = self.find_text(img_part)
        if len(prediction_groups[0]) == 0:
            return False
        prediction = prediction_groups[0][0][1]
        first_block_part = img_part[
                           int(prediction[0][1]):int(prediction[3][1]),
                           int(prediction[0][0]):int(prediction[1][0]),
                           ]
        font = self.predict_font(first_block_part)
        color = self.get_mean_color(first_block_part)
        img_part = self.clear_text(img_part, united_groups)

        centroid = self.find_polygon_centroid(united_groups)
        img_part = self.draw_text(
            img_part, new_text,
            centroid[0], centroid[1],
            self.get_box_height(prediction_groups[0][0]), color=color, font=font,
        )
        self.img[y:y + height, x:x + weight] = img_part
        return True

    def translate_text(self, x, y, weight, height, src="en", dest="uk"):
        img_part = self.img[y:y + height, x:x + weight]
        prediction_groups, united_groups = self.find_text(img_part)
        if len(prediction_groups[0]) == 0:
            return None
        block_with_text = img_part[
                           int(united_groups[0][1]):int(united_groups[3][1]),
                           int(united_groups[0][0]):int(united_groups[1][0]),
                           ]
        text = pytesseract.image_to_string(block_with_text, lang='eng+ukr',
                                           config="--oem 1 --psm 6").strip()
        text = self.translate_deepl(text, source_lang=src, target_lang=dest)
        return text

    def remove_text(self, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        prediction_groups, united_groups = self.find_text(img_part)
        if len(prediction_groups[0]) == 0:
            return False
        img_part = self.clear_text(img_part, united_groups)
        self.img[y:y + height, x:x + weight] = img_part
        return True

    def read_text(self, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        text = pytesseract.image_to_string(img_part, lang='eng+ukr',
                                           config="--oem 1 --psm 6").strip()
        return text

    def copy_text(self, x, y, weight, height):
        text = self.read_text(x, y, weight, height)
        pyperclip.copy(text)

    def draw_text(self, image, text, x, y, h, color=(255, 255, 255), font="Arial"):
        fontpath = f"data/fonts/{font}.ttf"
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(fontpath, int(h * 0.95))

        lines = text.split('\n')
        line_heights = []

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            line_heights.append(line_height)

        total_height = sum(line_heights) + (len(line_heights) - 1) * 10

        current_y = (img_pil.height - total_height) / 2

        for line, line_height in zip(lines, line_heights):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            start_x = x - text_width / 2
            draw.text((start_x, current_y), line, font=font, fill=color)
            current_y += line_height + 10
        return np.array(img_pil)
