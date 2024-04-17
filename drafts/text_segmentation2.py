import cv2
import numpy as np

def segment_text_better(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)  # Використання медіанного фільтру для зменшення шуму

    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opening)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 0.9 and 10 < w < 200:  # Уточнення умов для контурів
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    cv2.imshow('Original Image', image)
    cv2.imshow('Text Segmentation', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

segment_text_better('../data/img_1.png')
