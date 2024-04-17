import os
import time

import imageio
import numpy as np
from keras.models import load_model
from tqdm import tqdm

BASEDIR = "."
MODEL_DIR = os.path.join(BASEDIR, "saved_models")
model_name = os.path.join(MODEL_DIR, "font.model.01.keras")


def get_dir_paths(root):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if
            os.path.isdir(os.path.join(root, n))]


def get_file_paths(root):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if
            os.path.isfile(os.path.join(root, n))]


def normalize_image(image, pixel_depth):
    """
    Normalize image for all pixels. 0-255 -> 0.0-1.0
    """

    try:
        array = imageio.imread(image)
    except ValueError:
        raise

    return 1.0 - (array.astype(float)) / pixel_depth  # (1 - x) will make it white on black


def get_exact_image_size(image_data):
    """
    Find out where are the non white pixels in the image
    """
    mask = image_data != 0.0
    coords = np.argwhere(mask)

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1

    return x_min, x_max, y_min, y_max


def load_one_font_data(dir, image_size, pixel_depth,
                       verbose=False,
                       min_nb_images=1):
    """
    Function to load all data for a single font
    We will load:
    - 1x1 images containing 1 character
    - 3x1 images containing 3 characters
    Note:
    - For pictures, we typically describe them by width x height.
    - For 2D Matrix, we typically describe them by row x colum.
    This explains why we talk about a 1x3 matrix for a 3x1 image.
    """

    if verbose:
        print(dir)

    image_files = get_file_paths(dir)

    out_images = []

    image_index = 0
    for image in image_files:
        try:
            image_data_all_channels = normalize_image(image, pixel_depth)
            image_data = image_data_all_channels[:, :, 0]

            # Images proportions can be 1x1 or 3x1
            if image_data.shape != (image_size, image_size):
                # Process 3x1 image
                if image_data.shape != (image_size, 3 * image_size):
                    raise Exception(f'Unexpected image shape: {str(image_data.shape)}')
                else:
                    # We consider 3 x 1 image as a kind of elongated font "dough" in which
                    # we will cut 3 images:
                    # - a center image
                    # - a left image that will start where the first pixel is actually drawn
                    # - a right image that will end where the last pixel is actually drawn

                    # Images has: width:3 x height:1
                    x_min, x_max, y_min, y_max = get_exact_image_size(image_data)

                    # Center Image
                    # Find the first and last column to use in the 3 x 1 image
                    center_min_j = max(y_min, image_size * 1)
                    center_max_j = min(y_max, image_size * 2)

                    center_image = np.zeros((image_size, image_size))
                    center_image[x_min:x_max,
                    center_min_j - image_size:center_max_j - image_size] = image_data[x_min:x_max,
                                                                           center_min_j:center_max_j]
                    out_images.append(center_image)

                    # Left Image
                    left_min_j = max(y_min - 2, 0)  # Keep a 2 pixels margin on the left
                    left_max_j = left_min_j + image_size

                    left_image = np.zeros((image_size, image_size))
                    left_image[x_min:x_max, :] = image_data[x_min:x_max, left_min_j:left_max_j]
                    out_images.append(left_image)

                    # Right Image
                    right_max_j = min(y_max + 2,
                                      image_size * 3)  # Keep a 2 pixels margin on the right
                    right_min_j = right_max_j - image_size

                    right_image = np.zeros((image_size, image_size))
                    right_image[x_min:x_max, :] = image_data[x_min:x_max, right_min_j:right_max_j]
                    out_images.append(right_image)

            else:
                # Process 1 x 1 image
                out_images.append(image_data)
        except Exception as error:
            print(error)
            print('Skipping because of not being able to read: ', image)

    # Put all of our output images in a numpy array.
    # Note: There is probably a more efficient way to do that
    nb_output_images = len(out_images)
    dataset = np.ndarray(shape=(nb_output_images, image_size, image_size), dtype=np.float32)
    for image_index in range(nb_output_images):
        dataset[image_index, :, :] = out_images[image_index]

    if nb_output_images < min_nb_images:
        raise Exception(f'Fewer images than expected: {nb_output_images} < {min_nb_images}')

    if verbose:
        print('Full dataset tensor: ', dataset.shape)
        print('Mean: ', np.mean(dataset))
        print('Standard deviation: ', np.std(dataset))

    return dataset


def transform_all_font_data(pathnames, image_size, pixel_depth,
                            verbose=False,
                            min_nb_images=1):
    """
    Load data (images and labels) of all fonts
    """

    result = {}  # The font names will be the keys of dict. The values are all the images for that font
    for full_filepath in tqdm(pathnames):
        filename = os.path.basename(full_filepath)
        dataset = load_one_font_data(full_filepath, image_size, pixel_depth,
                                     verbose=verbose,
                                     min_nb_images=min_nb_images)
        result[filename] = dataset
    return result


def finalize_image_packaging(all_data, image_size):
    """
    Produce clean image data and labels. Also keep the list of font names.
    """

    images = []
    labels = []
    font_names = []

    nb_fonts = len(all_data)

    for label, font_name in enumerate(all_data):
        letter_data = all_data[font_name]

        # All images
        images.extend(letter_data)
        # all labels (integer value)
        labels.extend([label] * len(letter_data))
        # Keep the font names
        font_names.append(font_name)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, font_names


BASEDIR = "."

FONT_DATA_DIR = os.path.join(BASEDIR, "data", "font_data")

image_size = 50
pixel_depth = 255.0

font_data_paths = get_dir_paths(FONT_DATA_DIR)[:10]

all_data = transform_all_font_data(font_data_paths, image_size, pixel_depth, verbose=False)

images, labels, font_names = finalize_image_packaging(all_data, image_size)
# Y_val = to_categorical(val_y, num_classes=nb_fonts)


model = load_model(model_name)

start = time.time()
Y_pred = model.predict(images[:1])
print(time.time() - start)
Y_pred_classes = np.argmax(Y_pred, axis=1)

print(Y_pred)
print(Y_pred_classes)
