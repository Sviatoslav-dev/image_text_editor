import time

import keras_ocr
import matplotlib.pyplot as plt


# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images
images = [keras_ocr.tools.read('../data/img_1.png')]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)

# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for predictions in prediction_groups:
    keras_ocr.tools.drawAnnotations(image=images[0], predictions=predictions, ax=axs)

plt.show()
