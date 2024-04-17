import keras
import matplotlib.pyplot as plt
import itertools
import os
from pathlib import Path
from platform import python_version

import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, \
    Input, MaxPooling2D
from keras.models import load_model, Model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from numpy import repeat
from six.moves import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


print(python_version())

print(tf.__version__)

seed = 128
np.random.seed(seed)

BASEDIR = "."

# Input
PICKLE_DIR = os.path.join(BASEDIR, "pickles")
DATAFILE = os.path.join(PICKLE_DIR, 'font.pickle')

# Output
MODEL_DIR = os.path.join(BASEDIR, "saved_models")
# model_name = os.path.join(MODEL_DIR, "font.model.01.h5")
model_name = os.path.join(MODEL_DIR, "font.model.01.keras")

# Create Model directory
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

with open(DATAFILE, 'rb') as file:
    data_dict = pickle.load(file)

# Show the actual font names in the dataaz
output_labels = data_dict['font_names']

print("Fonts")
for label, font_name in enumerate(output_labels):
    print(f"{label}: {font_name}")

image_size = 50  # Pixel width and height
pixel_depth = 255.0  # Number of levels per pixel

nb_fonts = len(output_labels)

X = data_dict['images']
y = data_dict['labels']

images_per_font = X.shape[0] / nb_fonts

# Construct training set and validation set
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=seed)

# Training set
print(train_X.shape, train_y.shape)

# Validation set
print(val_X.shape, val_y.shape)

print(images_per_font)

# Reshape images to add the channel (height = 50px, width = 50px , channel = 1)
X_train = train_X.reshape(-1, image_size, image_size, 1)
X_val = val_X.reshape(-1, image_size, image_size, 1)


def plot_sample(image, axs):
    axs.imshow(image.reshape(image_size, image_size), cmap="gray")


def display_images(X, Y):
    """
    This function shows images with their real labels
    Presentation is rows of 5 images
    """

    fig = plt.figure(figsize=(13, 20))
    fig.subplots_adjust(hspace=1.1, wspace=0.1,
                        left=0, right=1, bottom=0, top=1)
    nb_pictures = len(X)
    nb_per_row = 5
    nb_of_row = (nb_pictures - 1) // nb_per_row + 1

    for i in range(nb_pictures):
        ax = fig.add_subplot(nb_of_row, nb_per_row, i + 1, xticks=[], yticks=[])
        plot_sample(X[i].reshape((image_size, image_size)), ax)
        ax.set_title("{}".format(output_labels[Y[i]]))
    # plt.show()


display_images(X_train[:50], train_y[:50])

display_images(X_val[:50], val_y[:50])

print(train_y.shape, val_y.shape)

# Encode labels to one hot vectors
Y_train = to_categorical(train_y, num_classes=nb_fonts)
print(val_y)
raise Exception
Y_val = to_categorical(val_y, num_classes=nb_fonts)

print(X_train.shape, X_val.shape)

print(Y_train.shape, Y_val.shape)

keras.backend.clear_session()


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
optimizer = Adam(learning_rate=0.0001, decay=0.0)

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

epochs = 2  # Turn epochs to ...
batch_size = 128
history = model.fit(datagen.flow(repeat(X_train, epochs, axis=0), repeat(Y_train, epochs, axis=0),
                                 batch_size=batch_size),
                    epochs=epochs, validation_data=(X_val, Y_val),
                    verbose=1, steps_per_epoch=X_train.shape[0] // batch_size)

# From now on, we will save the model with the best accuracy
checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', verbose=1, save_best_only=True,
                             mode='max')

best_epoch_results = ShowBestEpochResult(on_param='val_accuracy',
                                         show_params=['val_accuracy', 'accuracy', 'val_loss',
                                                      'loss'],
                                         mode='max')

callbacks_list = [learning_rate_reduction, checkpoint, best_epoch_results]

epochs = 5
batch_size = 128
history = model.fit(datagen.flow(repeat(X_train, epochs, axis=0), repeat(Y_train, epochs, axis=0),
                                 batch_size=batch_size),
                    epochs=epochs, validation_data=datagen.flow(X_val, Y_val),
                    verbose=1, steps_per_epoch=X_train.shape[0] // batch_size,
                    callbacks=callbacks_list)


def plot_loss(hist, name, plt):
    loss = hist['loss']
    val_loss = hist['val_loss']

    plt.plot(loss, "--", linewidth=3, label="train:" + name)
    plt.plot(val_loss, linewidth=3, label="val:" + name)


def plot_accuracy(hist, name, plt):
    loss = hist['accuracy']
    val_loss = hist['val_accuracy']

    plt.plot(loss, "--", linewidth=3, label="train:" + name)
    plt.plot(val_loss, linewidth=3, label="val:" + name)


plt.figure(figsize=(15, 10))
plot_loss(history.history, "model", plt)
plt.legend()
plt.grid()
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("log loss")
plt.show()

plt.figure(figsize=(15, 15))
plot_accuracy(history.history, "model", plt)
plt.legend()
plt.grid()
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("log accuracy")
plt.show()

# deletes the existing model
del model

# Load last saved model, ie the one with the best accuracy
model = load_model(model_name)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(15, 13))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Predict the values from the validation dataset
Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(nb_fonts))

print("Y_pred shape: ", Y_pred.shape)  # Prediction for each image in the validaset set
print("Y_pred_classes shape: ",
      Y_pred_classes.shape)  # Index of the font prediction for each image in the val. set
print("Y_true shape: ", Y_true.shape)  # Index of the actual font for each image in the val. set
print("confusion_mtx shape: ", confusion_mtx.shape)  # Values in the confusion matrix shown above

print(Y_pred_classes[0:12])

print(X_val.shape)

# Display some error results

# Errors are difference between predicted labels and true labels. Error[i] is True if this is an error
errors = (Y_pred_classes - Y_true != 0)

# Y_pred_errors: List of Images in the validation set where prediction was bad
X_val_errors = X_val[errors]

# Y_true_errors: Index of the actual font in the validation set where prediction was bad
Y_true_errors = Y_true[errors]

# Y_pred_classes_errors: The list of indexes for font predictions that were bad
Y_pred_classes_errors = Y_pred_classes[errors]

# Y_pred_errors: Prediction for each font in the validation set where prediction was bad
Y_pred_errors = Y_pred[errors]


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows images with their predicted and real labels"""

    fig = plt.figure(figsize=(13, 15))
    fig.subplots_adjust(hspace=0.13, wspace=0.2, left=0, right=1, bottom=0, top=1)
    nb_pictures = len(errors_index)
    nb_per_row = 5
    nb_of_row = (nb_pictures - 1) // nb_per_row + 1

    for i in range(nb_pictures):
        ax = fig.add_subplot(nb_of_row, nb_per_row, i + 1, xticks=[], yticks=[])
        error = errors_index[i]
        plot_sample((img_errors[error]).reshape((image_size, image_size)), ax)
        ax.set_title("Pred: {}\nTrue : {}".format(output_labels[pred_errors[error]],
                                                  output_labels[obs_errors[error]]))
    plt.show()


# Probabilities of the wrong predicted numbers. Prob for the bad choice
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

# Predicted probabilities of the true values in the error set. Prob for the good choice
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_delta_errors = np.argsort(delta_pred_true_errors)

# Top errors
most_important_errors = sorted_delta_errors[-25:]

# Show the top errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


def saliency_map(input_image, model, label, image_size, num_classes):
    '''
    Generates the saliency map of a given gray scale image.

    Args:
        image -- picture that the model will classify
        model (keras Model) -- classifier
        label (int) -- ground truth label of the image
        image_size (int)  -- Image size in pixels
        num_classes (int) -- number of classes for label
    '''

    input_image = input_image.reshape(image_size, image_size)

    # Add an additional dimension (for the batch)
    image = np.expand_dims(input_image, axis=0)

    # Define the expected output array by one-hot encoding the label
    # The length of the array is equal to the number of classes
    expected_output = tf.one_hot([label] * image.shape[0], num_classes)

    with tf.GradientTape() as tape:
        # cast image to float
        inputs = tf.cast(image, tf.float32)

        # watch the input pixels
        tape.watch(inputs)

        # generate the predictions
        predictions = model(inputs)

        # Calculate the loss
        loss = tf.keras.losses.categorical_crossentropy(expected_output, predictions)

    # get the gradients of the loss with respect to the model's input image
    gradients = tape.gradient(loss, inputs)

    # generate the grayscale tensor
    grayscale_tensor = tf.abs(gradients[0])

    # The max value in the grayscale tensor will be pushed to 1.0
    # The min value will be pushed to 0.
    normalized_tensor = (grayscale_tensor - tf.reduce_min(grayscale_tensor)) / \
                        (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor))

    # Mix the saliency map with the original image
    mixed_image = input_image * 0.5 + normalized_tensor * 0.5

    return mixed_image.numpy()


def display_images_with_saliency_map(X, Y):
    """ This function shows Saliency maps for 25 images with their real labels"""

    fig = plt.figure(figsize=(13, 13))
    fig.subplots_adjust(hspace=0.1, wspace=0.2, left=0, right=1, bottom=0, top=1)
    nb_pictures = 25
    m = 0

    for i in range(nb_pictures):
        image_with_saliency = saliency_map(X[i], model, Y[i], image_size, nb_fonts)

        if np.isnan(image_with_saliency[0, 0]):
            # TODO Investigate when this condition occurs exactly
            continue

        ax = fig.add_subplot(nb_pictures // 5, 5, m + 1, xticks=[], yticks=[])
        plot_sample(image_with_saliency.reshape((image_size, image_size)), ax)
        ax.set_title("{}".format(output_labels[Y[m]]))

        m += 1
    plt.show()


# Do a visualization of which pixels the model is focusing on when making its predictions
display_images_with_saliency_map(X_val[:25], val_y[:25])

# Debugging information
print("Validation shape: ", errors.shape)
print("X_val_errors shape: ", X_val_errors.shape)
print("Y_true_errors shape: ", Y_true_errors.shape)
print("Y_pred_classes_errors shape: ", Y_pred_classes_errors.shape)
print("Y_pred_errors shape: ", Y_pred_errors.shape)

# More Debugging information
print("Y_pred_errors_prob shape: ", Y_pred_errors_prob.shape)
print("true_prob_errors shape: ", true_prob_errors.shape)
print("delta_pred_true_errors shape: ", delta_pred_true_errors.shape)
print("sorted_delta_errors shape: ", sorted_delta_errors.shape)
print("most_important_errors shape: ", most_important_errors.shape)

nb_errors = X_val_errors.shape[0]
nb_val_samples = Y_pred.shape[0]
accuracy = 1.0 - (nb_errors / nb_val_samples)

print("Validation Accuracy:", accuracy)
print(f"Number of errors: {nb_errors} from {nb_val_samples} predictions on validation set.")
