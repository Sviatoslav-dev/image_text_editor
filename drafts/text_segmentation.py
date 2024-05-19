import cv2
import numpy as np


def segmentate_text(image):
    # Перетворення в градації сірого та використання адаптивного порога
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Застосування розмиття для зменшення шуму
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 3, 2)
    cv2.imshow('adaptive_thresh', adaptive_thresh)

    # # Морфологічні операції для з'єднання компонентів тексту
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    connected_text = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('connected_text', connected_text)

    # Виявлення контурів
    contours, _ = cv2.findContours(connected_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_mask = np.zeros_like(adaptive_thresh)  # Створення порожньої маски для тексту

    # Заповнення маски білим кольором в місцях, де виявлено текст
    cv2.drawContours(text_mask, contours, -1, (255), thickness=cv2.FILLED)
    # mask = text_mask - connected_text
    return text_mask


def get_mean_color(image):
    mask = segmentate_text(image)
    mean = cv2.mean(image, mask)
    return int(mean[0]), int(mean[1]), int(mean[2])


if __name__ == "__main__":
    def create_full_text_mask(image_path):
        # Завантаження зображення
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Image not found.")
            return

        # Відображення результатів
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Adaptive Threshold', adaptive_thresh)
        # cv2.imshow('Connected Text Regions', connected_text)
        # cv2.imshow('Text Mask', text_mask)
        print(get_mean_color(image))
        text_img = segmentate_text(image)
        cv2.imshow('New Text Mask', text_img)
        cv2.imshow('New Text', cv2.bitwise_and(image, image, mask=text_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Виклик функції з шляхом до вашого зображення
    create_full_text_mask('../data/img_6.png')
