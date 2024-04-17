import matplotlib.pyplot as plt
import itertools
import os
from pathlib import Path
from platform import python_version
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, \
    Input, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizer_v2.adam import Adam
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from numpy import asarray
from six.moves import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import repeat

from font_from_image_main.fonts import fonts

seed = 128
np.random.seed(seed)

MODEL_DIR = "saved_models"
# model_name = os.path.join(MODEL_DIR, "font.model2.01.h5")
model_name = os.path.join(MODEL_DIR, "font.model.02.keras")

# Create Model directory
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Show the actual font names in the dataaz

print("Fonts")
for label, font_name in enumerate(fonts):
    print(f"{label}: {font_name}")

image_size = 50  # Pixel width and height
pixel_depth = 255.0  # Number of levels per pixel

nb_fonts = len(fonts)


def load_data(path):
    X = []
    y = []
    i = 0
    for font in output_labels:
        for file_name in os.listdir(f"{path}/{font}"):
            image = Image.open(f"{path}/{font}/{file_name}")
            X.append(asarray(image))
            y.append(i)
        i += 1
    return np.array(X), np.array(y)


X, y = load_data("../data/fonts_data")
print(X.shape)
print(y.shape)

images_per_font = X.shape[0] / nb_fonts

# Construct training set and validation set
X_train, X_val, train_y, val_y = train_test_split(X[:, :, :, 0:1], y, test_size=0.2,
                                                  random_state=seed)

# Training set
print(X_train.shape, train_y.shape)

# Validation set
print(X_val.shape, val_y.shape)

print(images_per_font)


# Reshape images to add the channel (height = 50px, width = 50px , channel = 1)
# X_train = train_X.reshape(-1, image_size, image_size, 1)
# X_val = val_X.reshape(-1, image_size, image_size, 1)

# print(X_train.shape)
# print(X_val.shape)


def plot_sample(image, axs):
    axs.imshow(image.reshape(image_size, image_size), cmap="gray")


print(train_y.shape, val_y.shape)

# Encode labels to one hot vectors
Y_train = to_categorical(train_y, num_classes=nb_fonts)
print(val_y)
Y_val = to_categorical(val_y, num_classes=nb_fonts)

print(X_train.shape, X_val.shape)

print(Y_train.shape, Y_val.shape)

tf.keras.backend.clear_session()


def get_model_cnn1():
    # Set the CNN model using Keras functional API
    input_img = Input(shape=(image_size, image_size, 1))

    x = Conv2D(32, (5, 5), padding='same')(input_img)  # Conv2D 1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (5, 5), padding='same')(x)  # Conv2D 2
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (5, 5))(x)  # Conv2D 3
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)  # Conv2D 4
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3))(x)  # Conv2D 5
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3))(x)  # Conv2D 6
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

    return model


# select a model
model = get_model_cnn1()

model.summary()

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=[0.9, 1.11],  # Randomly zoom image
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images


# Keras callback to the result of the best epoch in the model
# Intended to work with ModelCheckpoint callback.
class ShowBestEpochResult(tf.keras.callbacks.Callback):
    def __init__(self, on_param='val_loss', show_params=['val_loss', 'loss'], mode='min'):
        self.on_param = on_param
        self.show_params = show_params
        self.mode = mode

    def on_train_begin(self, logs=None):
        # The number of epoch at which the best value is found.
        self.best_epoch = 0

        # The best value found for the on_param.
        self.best = np.Inf
        if self.mode == 'max':
            self.best = np.NINF

        # The values on best epoch for each param in show_param.
        self.best_values = {}

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.on_param)
        save_as_best = (self.mode == 'min' and current < self.best) or \
                       (self.mode == 'max' and current > self.best)

        if save_as_best:
            self.best = current
            for param in self.show_params:
                self.best_values[param] = logs.get(param)
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        print(f"BestEpochResult. Epoch: {self.best_epoch + 1},", end=" ")

        for param in self.best_values:
            print(f"{param}: {self.best_values[param]:.5f},", end=" ")
        print(" ")


# Define the optimizer. Adam is generally my best bet.
optimizer = Adam(lr=0.0001, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Lower the learning rate if accuracy stops improving.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=8,
                                            verbose=1,
                                            factor=0.8,
                                            min_delta=1e-7,
                                            min_lr=1e-7)

# Fit the model with data Augmentation
# I like to start a few epoch with a small batch size because:
# - This basically makes the learning rate larger. https://openreview.net/pdf?id=B1Yy1BxCZ#:~:text=Increasing%20the%20batch%20size%20during%20training%20achieves%20similar%20results%20to,twice%20to%20illustrate%20the%20variance.
# - This covers to 0 to 80% of the accuracy curve, cleaning the graphs
print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)

# epochs = 2  # Turn epochs to ...
# batch_size = 128
# history = model.fit(
#     x=X_train,
#     y=Y_train,
#     batch_size=None,
#     epochs=1,
#     verbose=1,
#     callbacks=None,
#     validation_split=0.0,
#     validation_data=(X_val, Y_val),
#     shuffle=True,
#     class_weight=None,
#     sample_weight=None,
#     initial_epoch=0,
#     steps_per_epoch=None,
#     validation_steps=None,
#     validation_batch_size=None,
#     validation_freq=1,
# )
# # history = model.fit(
# #     datagen.flow(repeat(X_train, epochs, axis=0), repeat(Y_train, epochs, axis=0),
# #                  batch_size=batch_size),
# #     epochs=epochs, validation_data=(X_val, Y_val),
# #     verbose=1, steps_per_epoch=X_train.shape[0] // batch_size
# # )
#
# # From now on, we will save the model with the best accuracy
checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', verbose=1, save_best_only=True,
                             mode='max')

best_epoch_results = ShowBestEpochResult(on_param='val_accuracy',
                                         show_params=['val_accuracy', 'accuracy', 'val_loss',
                                                      'loss'],
                                         mode='max')

callbacks_list = [learning_rate_reduction, checkpoint, best_epoch_results]

epochs = 2
batch_size = 16
# history = model.fit(datagen.flow(repeat(X_train, epochs, axis=0), repeat(Y_train, epochs, axis=0),
#                                  batch_size=batch_size),
#                     epochs=epochs, validation_data=datagen.flow(X_val, Y_val),
#                     verbose=1, steps_per_epoch=X_train.shape[0] // batch_size,
#                     callbacks=callbacks_list)
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


# def plot_loss(hist, name, plt):
#     loss = hist['loss']
#     val_loss = hist['val_loss']
#
#     plt.plot(loss, "--", linewidth=3, label="train:" + name)
#     plt.plot(val_loss, linewidth=3, label="val:" + name)
#
#
# def plot_accuracy(hist, name, plt):
#     loss = hist['accuracy']
#     val_loss = hist['val_accuracy']
#
#     plt.plot(loss, "--", linewidth=3, label="train:" + name)
#     plt.plot(val_loss, linewidth=3, label="val:" + name)

model.save(model_name)

del model

# Load last saved model, ie the one with the best accuracy
# model = load_model(model_name)

# plt.figure(figsize=(15, 10))
# plot_loss(history.history, "model", plt)
# plt.legend()
# plt.grid()
# plt.yscale("log")
# plt.xlabel("epoch")
# plt.ylabel("log loss")
# plt.show()
#
# plt.figure(figsize=(15, 15))
# plot_accuracy(history.history, "model", plt)
# plt.legend()
# plt.grid()
# plt.yscale("log")
# plt.xlabel("epoch")
# plt.ylabel("log accuracy")
# plt.show()
