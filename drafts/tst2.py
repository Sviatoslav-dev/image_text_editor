import os
import pickle

import numpy as np
from pathlib import Path


seed = 128
np.random.seed(seed)

BASEDIR = "."

# Input
PICKLE_DIR = os.path.join(BASEDIR, "../font_from_image_main/pickles")
DATAFILE = os.path.join(PICKLE_DIR, 'font.pickle')

# Output
MODEL_DIR = os.path.join(BASEDIR, "../font_from_image_main/saved_models")
# model_name = os.path.join(MODEL_DIR, "font.model.01.h5")
model_name = os.path.join(MODEL_DIR, "font.model.01.keras")

# Create Model directory
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

with open(DATAFILE, 'rb') as file:
    data_dict = pickle.load(file)

print(data_dict)