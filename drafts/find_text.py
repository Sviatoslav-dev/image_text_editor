import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()


def annotate_text(image):
    predictions = pipeline.recognize([image])
    # keras_ocr.tools.drawAnnotations(image=image, predictions=predictions[0])
    return keras_ocr.tools.drawBoxes(image=image, boxes=predictions[0], boxes_format="predictions")


def find_text(image):
    predictions = pipeline.recognize([image])
    # keras_ocr.tools.drawAnnotations(image=image, predictions=predictions[0])
    return predictions
