import math

import cv2
import keras_ocr
import numpy as np
from matplotlib import pyplot as plt


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images
image = keras_ocr.tools.read('../data/img_1.png')

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize([image])

# example of a line mask for the word "Tuesday"
box = prediction_groups[0][0]
x0, y0 = box[1][0]
x1, y1 = box[1][1]
x2, y2 = box[1][2]
x3, y3 = box[1][3]

x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)

# masked = cv2.bitwise_and(image, image, mask=mask)
# plt.imshow(masked)
img_inpainted = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
plt.imshow(img_inpainted)
plt.show()


