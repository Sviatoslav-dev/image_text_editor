import numpy as np

from font_from_image_main.font_model import model
import imageio

pixel_depth = 255.0


def normalize_image(image, pixel_depth):
    """
    Normalize image for all pixels. 0-255 -> 0.0-1.0
    """

    # try:
    #     array = imageio.imread(image)
    # except ValueError:
    #     raise

    return 1.0 - (image.astype(float)) / pixel_depth  # (1 - x) will make it white on black


def get_exact_image_size(image_data):
    mask = image_data != 0.0
    coords = np.argwhere(mask)

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1

    return x_min, x_max, y_min, y_max


def load_one_font_data(image, image_size, pixel_depth):
    image_data_all_channels = normalize_image(image, pixel_depth)
    image_data = image_data_all_channels[:, :, 0]

    # Images proportions can be 1x1 or 3x1
    if image_data.shape != (image_size, image_size):
        image_data = image_data[:50, :50]

    # Put all of our output images in a numpy array.
    # Note: There is probably a more efficient way to do that
    dataset = np.ndarray(shape=(1, image_size, image_size), dtype=np.float32)
    dataset[0, :, :] = image_data

    return dataset


def predict_font(numpy_img):
    Y_pred = model.predict(load_one_font_data(numpy_img, 50, pixel_depth))
    return Y_pred
