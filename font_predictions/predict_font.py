import cv2
import numpy as np
from PIL import Image
from numpy import asarray

from font_from_image_main.font_model import model
from font_from_image_main.fonts import fonts


def predict_font(img: np.ndarray):
    height, weight, _ = img.shape
    k = height / 50
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_image.shape)
    new_weight = int(weight / k)
    resized_image = cv2.resize(gray_image, (new_weight, 50))
    print(resized_image.shape)
    resized_image = np.reshape(resized_image, (50, new_weight, 1))
    print(resized_image.shape)
    if resized_image.shape[1] > 50:
        resized_image = resized_image[:, :50]
    prediction = model.predict(np.array([resized_image]))[0]
    font_num = np.argmax(prediction)
    print(font_num)
    print(fonts[int(font_num)])
    return fonts[int(font_num)]


if __name__ == "__main__":
    image = Image.open("../data/fonts_data/Arial/text_3_2.png")
    img = asarray(image)
    predict_font(img)
