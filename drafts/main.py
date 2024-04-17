import time

import cv2
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('../data/img.png')
start = time.time()
text = pytesseract.image_to_string(img)
end = time.time()
print(text)
print(end-start)
