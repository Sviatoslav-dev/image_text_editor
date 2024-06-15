import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, \
    Input, MaxPooling2D
from keras.models import Model
from keras.optimizer_v2.adam import Adam
from keras.utils.np_utils import to_categorical
from numpy import asarray
from numpy import repeat
from sklearn.model_selection import train_test_split

from font_predictions.fonts import fonts

seed = 128
np.random.seed(seed)

model_dir = "saved_models"
model_name = os.path.join("saved_models", "font.model.03.keras")

Path(model_dir).mkdir(parents=True, exist_ok=True)

image_size = 50
pixel_depth = 255.0

nb_fonts = len(fonts)


def load_data(path):
    X = []
    y = []
    i = 0
    for font in fonts:
        for file_name in os.listdir(f"{path}/{font}"):
            image = Image.open(f"{path}/{font}/{file_name}")
            X.append(asarray(image))
            y.append(i)
        i += 1
    return np.array(X), np.array(y)


X, y = load_data("../data/fonts_data")

images_per_font = X.shape[0] / nb_fonts

X_train, X_val, train_y, val_y = train_test_split(X[:, :, :, 0:1], y, test_size=0.2,
                                                  random_state=seed)

Y_train = to_categorical(train_y, num_classes=nb_fonts)
Y_val = to_categorical(val_y, num_classes=nb_fonts)

tf.keras.backend.clear_session()


input_img = Input(shape=(image_size, image_size, 1))

x = Conv2D(32, (5, 5), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (5, 5), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(64, (5, 5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(128, (3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.3)(x)

x = Flatten()(x)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.4)(x)
output = Dense(nb_fonts, activation="softmax")(x)

model = Model(inputs=input_img, outputs=output)

print(model.summary())

optimizer = Adam(lr=0.0001, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)


epochs = 4
batch_size = 64
history = model.fit(
    x=repeat(X_train, epochs, axis=0),
    y=repeat(Y_train, epochs, axis=0),
    batch_size=None,
    epochs=epochs,
    verbose=1,
    callbacks=None,
    validation_split=0.0,
    validation_data=(X_val, Y_val),
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=X_train.shape[0] // batch_size,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
)

model.save(model_name)
