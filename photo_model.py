import cv2

from base_model import BaseImageModel
import pyperclip


class PhotoModel(BaseImageModel):
    def __init__(self):
        super().__init__()
        self.img = None

    def read_image(self, path):
        self.img = cv2.imread(path, 1)
        if self.img.shape[0] / self.img.shape[1] < 0.76:
            self.img_width = 1100
            self.img_height = int(self.img_width * self.img.shape[0] / self.img.shape[1])
        else:
            self.img_height = 700
            self.img_width = int(self.img_height * self.img.shape[1] / self.img.shape[0])

        self.img = cv2.resize(self.img, (self.img_width, self.img_height))

    def replace_text(self, new_text, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        # numpy_image = annotate_text(numpy_image)
        prediction_groups, united_groups = self.find_text(img_part)
        prediction = prediction_groups[0][0][1]
        first_block_part = img_part[
            int(prediction[0][1]):int(prediction[3][1]),
            int(prediction[0][0]):int(prediction[1][0]),
            ]
        font = self.predict_font(first_block_part)
        color = self.get_mean_color(first_block_part)
        print("color: ", color)
        print("font: ", font)
        for box in prediction_groups[0]:
            img_part = self.clear_text(img_part, box[1])

        centroid = self.find_polygon_centroid(united_groups)
        img_part = self.draw_text(
            img_part, new_text,
            centroid[0], centroid[1],
            self.get_box_height(prediction_groups[0][0]), color=color, font=font,
        )
        self.img[y:y + height, x:x + weight] = img_part

    def remove_text(self, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        prediction_groups, united_groups = self.find_text(img_part)
        img_part = self.clear_text(img_part, prediction_groups[0][0][1])
        self.img[y:y + height, x:x + weight] = img_part

    def copy_text(self, x, y, weight, height):
        img_part = self.img[y:y + height, x:x + weight]
        prediction_groups, united_groups = self.find_text(img_part)
        lines = self.group_text_by_lines(prediction_groups[0])

        text = '\n'.join(lines)

        pyperclip.copy(text)
        print(prediction_groups)
