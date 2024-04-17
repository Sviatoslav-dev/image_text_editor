import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw


def get_box_height(box):
    return int((box[1][3][1] - box[1][0][1]) * 0.8)


def draw_text(image, text, x, y, h, color=(255, 255, 255), font="Arial"):
    print("xy", x, y)
    print("h = ", h)
    # thickness = 2
    # text_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, h, thickness)
    # return cv2.putText(
    #     image, text, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #     text_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    # )

    fontpath = f"data/fonts/{font}.ttf"
    font = ImageFont.truetype(fontpath, int(h * 1.3))
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((int(x) + 5, int(y) - 5), text, font=font, fill=color)
    img = np.array(img_pil)
    return img
