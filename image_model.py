import cv2
import numpy as np
import pyperclip
import pytesseract
import requests
from PIL import ImageFont, Image, ImageDraw

from base_model import BaseImageModel


def translate_deepl(text, target_lang='UK'):
    api_key = '5d781a78-1aef-4f04-be1d-ba831c57c183:fx'
    url = 'https://api-free.deepl.com/v2/translate'

    params = {
        'auth_key': api_key,
        'text': text,
        'target_lang': target_lang
    }

    response = requests.post(url, data=params)

    if response.status_code == 200:
        return response.json()['translations'][0]['text']
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


class ImageModel(BaseImageModel):
    def __init__(self):
        super().__init__()
        self.img = None
        self.translate_option = ("en", "uk")

    def read_image(self, path):
        self.img = cv2.imread(path, 1)
        if self.img.shape[0] / self.img.shape[1] < 0.76:
            self.img_width = 1100
            self.img_height = int(self.img_width * self.img.shape[0] / self.img.shape[1])
        else:
            self.img_height = 700
            self.img_width = int(self.img_height * self.img.shape[1] / self.img.shape[0])

        self.img = cv2.resize(self.img, (self.img_width, self.img_height))
        self.start_image = np.copy(self.img)

    def replace_text(self, new_text, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        # numpy_image = annotate_text(numpy_image)
        prediction_groups, united_groups = self.find_text(img_part)
        if len(prediction_groups[0]) == 0:
            return False
        prediction = prediction_groups[0][0][1]
        # for pp in prediction_groups[0]:
        #     for i in range(5):
        #         block = img_part[
        #            int(pp[1][0][1]):int(pp[1][3][1]),
        #            int(pp[1][0][0]):int(pp[1][1][0]),
        #         ]
        #         font = self.predict_font(block)
        #         print("ffffffffffffff: ", font)
        first_block_part = img_part[
                           int(prediction[0][1]):int(prediction[3][1]),
                           int(prediction[0][0]):int(prediction[1][0]),
                           ]
        font = self.predict_font(first_block_part)
        color = self.get_mean_color(first_block_part)
        print("color: ", color)
        print("font: ", font)
        # for box in prediction_groups[0]:
        #     img_part = self.clear_text(img_part, box[1])
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
        print("text: ", text)
        text = translate_deepl(text)
        return text

    def remove_text(self, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        prediction_groups, united_groups = self.find_text(img_part)
        if len(prediction_groups[0]) == 0:
            return False
        img_part = self.clear_text(img_part, prediction_groups[0][0][1])
        self.img[y:y + height, x:x + weight] = img_part
        return True

    def read_text(self, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        # prediction_groups, united_groups = self.find_text(img_part)
        # lines = self.group_text_by_lines(prediction_groups[0])
        #
        # text = '\n'.join(lines)
        text = pytesseract.image_to_string(img_part, lang='eng+ukr',
                                           config="--oem 1 --psm 6").strip()

        return text

    def copy_text(self, x, y, weight, height):
        text = self.read_text(x, y, weight, height)
        pyperclip.copy(text)

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
        # font = ImageFont.truetype(fontpath, int(h * 1.3))
        # img_pil = Image.fromarray(image)
        # draw = ImageDraw.Draw(img_pil)
        #
        # ascent, descent = font.getmetrics()
        # text_bbox = draw.textbbox((0, 0), text, font=font)
        # text_width = text_bbox[2] - text_bbox[0]
        # # text_height = text_bbox[3] - text_bbox[1]
        # text_height = ascent - descent
        # text_height *= 1.5
        #
        # start_x = x - text_width // 2
        # start_y = y - text_height // 2
        # draw.text((start_x, start_y), text, font=font, fill=color)
        # img = np.array(img_pil)
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
