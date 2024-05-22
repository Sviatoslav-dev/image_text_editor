import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from font_predictions.fonts import fonts


def add_noise(img):
    image_np = np.array(img)

    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (img.height, img.width, 3))  # для RGB зображення

    noisy_image = np.clip(image_np + gaussian, 0, 255)

    noisy_image = Image.fromarray(np.uint8(noisy_image))
    return noisy_image


def create_text_image(text, font_path, font_size, num):
    font_color = random.randint(0, 255)
    image = Image.new('RGB', (150, 50),
                      color=(255 - font_color, 255 - font_color, 255 - font_color))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("../data/fonts/" + font_path + ".ttf", font_size)

    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]

    x = (image.width - text_width) / 2
    y = (image.height - text_height) / 2

    draw.text((x, y), text, font=font, fill=(font_color, font_color, font_color))

    image = add_noise(image)

    image.crop((0, 0, 50, 50)).save("../data/fonts_data/" + font_path + f"/text_{num}_1.png")
    image.crop((50, 0, 100, 50)).save("../data/fonts_data/" + font_path + f"/text_{num}_2.png")
    image.crop((100, 0, 150, 50)).save("../data/fonts_data/" + font_path + f"/text_{num}_3.png")


fonts_count = 5000

for font_name in fonts:
    print(font_name)
    os.makedirs("../data/fonts_data/" + font_name)
    for i in range(fonts_count // 3):
        text = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in
                       range(random.randint(5, 10)))
        create_text_image(text, font_name, random.randint(30, 50), num=i)
