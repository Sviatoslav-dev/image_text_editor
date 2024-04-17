import time

import numpy as np
import cv2 as cv

import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()


def annotate_text(images):
    # predictions = pipeline.recognize(images)
    # keras_ocr.tools.drawAnnotations(image=image, predictions=predictions[0])
    frames = []
    for i, image in enumerate(images):
        print(i)
        predictions = pipeline.recognize([image])
        # frames.append(
        #     keras_ocr.tools.drawBoxes(image=image, boxes=predictions[0], boxes_format="predictions")
        # )
        return predictions

cap = cv.VideoCapture('../data/video_with_text5.mp4')

frames = []

ret, frame = cap.read()
# box = annotate_text([frame])[0]
bbox = cv.selectROI("Tracking", frame)
# tracker = cv.legacy.TrackerMOSSE_create()
tracker = cv.legacy.TrackerCSRT_create()
tracker.init(frame, bbox)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle(img, (x, y), ((x+w), (y+h)), (225, 0, 225), 3, 1)
    cv.putText(img, "Tracking", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)



while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    success, box = tracker.update(frame)
    drawBox(frame, box)

    if success:
        pass
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame = annotate_text(frame)
    # frames.append(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    time.sleep(0.02)

print("here")
print(len(frames))
frames = annotate_text(frames)
print("annotated")


for frame in frames:
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    time.sleep(0.02)

cap.release()
cv.destroyAllWindows()
