import cv2
import numpy as np

net = cv2.dnn.readNet('frozen_east_text_detection.pb')


def decode_predictions(scores, geometry, conf_threshold=0.5):
    numRows, numCols = scores.shape[2:4]
    boxes = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < conf_threshold:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            boxes.append((startX, startY, endX - startX, endY - startY))
            confidences.append(float(scoresData[x]))

    return boxes, confidences


def highlight_text(frame):
    origH, origW = frame.shape[:2]
    newW, newH = 320, 320
    rW = origW / float(newW)
    rH = origH / float(newH)

    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True,
                                 crop=False)
    net.setInput(blob)

    scores, geometry = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

    boxes, confidences = decode_predictions(scores, geometry)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_boxes = [boxes[i] for i in indices] if len(indices) > 0 else []

    scaled_boxes = [(int(x * rW), int(y * rH), int(w * rW), int(h * rH)) for (x, y, w, h) in
                    detected_boxes]

    for x, y, w, h in scaled_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def convert_box_to_polygon(box):
    startX, startY, sizeX, sizeY = box
    endX = startX + sizeX
    endY = startY + sizeY

    top_left = (startX, startY)
    top_right = (endX, startY)
    bottom_right = (endX, endY)
    bottom_left = (startX, endY)

    polygon = np.array([top_left, top_right, bottom_right, bottom_left])

    return polygon


def find_text_by_east(frame):
    origH, origW = frame.shape[:2]
    newW, newH = 320, 320
    rW = origW / float(newW)
    rH = origH / float(newH)

    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True,
                                 crop=False)
    net.setInput(blob)

    scores, geometry = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

    boxes, confidences = decode_predictions(scores, geometry)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_boxes = [boxes[i] for i in indices] if len(indices) > 0 else []

    scaled_boxes = [(int(x * rW), int(y * rH), int(w * rW), int(h * rH)) for (x, y, w, h) in
                    detected_boxes]

    return [convert_box_to_polygon(box) for box in scaled_boxes]


if __name__ == '__main__':
    cap = cv2.VideoCapture('path_to_your_video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_text = highlight_text(frame)

        cv2.imshow('Video with Text Highlight', frame_with_text)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
