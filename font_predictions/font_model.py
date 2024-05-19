import os

import numpy as np
from PIL import Image
from keras.models import load_model
from numpy import asarray

# BASEDIR = "."
BASEDIR = "."
MODEL_DIR = os.path.join(BASEDIR, "font_predictions/saved_models")

model_name = os.path.join(MODEL_DIR, "font.model.02.keras")

model = load_model(model_name)

if __name__ == "__main__":
    image = Image.open("./data/fonts_data/timesnewromanpsmt/text_3_2.png")
    X = asarray(image)
    print(X[:, :, 0:1].shape)
    print(model.predict(np.array([X])[:, :, :, 0:1]))
