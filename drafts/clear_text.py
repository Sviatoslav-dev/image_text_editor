import math

import cv2
import numpy as np


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return x_mid, y_mid


def clear_text(image, prediction_group=None, box=None):
    box = prediction_group[0] if box is None else box
    x0, y0 = box[1][0]
    x1, y1 = box[1][1]
    x2, y2 = box[1][2]
    x3, y3 = box[1][3]

    x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
    x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
    thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)

    img_inpainted = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    return img_inpainted
