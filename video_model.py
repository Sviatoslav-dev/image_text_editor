import cv2
import numpy as np
import pyperclip
import pytesseract

from base_editor.base_model import BaseImageModel


class VideoModel(BaseImageModel):
    def __init__(self, video_path):
        super().__init__()
        self.video_capture = cv2.VideoCapture()

        self.fps = None
        self.setup_video(video_path)
        self.frame_num = 0
        self.frames = []
        self.frames_count = self.get_frames_count()
        self.read_all_frames()
        self.translate_option = ("en", "uk")

    def read_all_frames(self):
        self.frames = []
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            self.frames.append(frame)

    def setup_video(self, path):
        self.video_capture.open(path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

    def replace_text(self, new_text, x, y, weight, height):
        current_frame = self.get_current_frame()

        img_part = current_frame[y:y + height, x:x + weight]
        predictions, united_groups = self.find_text(img_part)
        if len(predictions[0]) == 0:
            return False
        prediction = predictions[0][0][1]
        first_block_part = img_part[
                           int(prediction[0][1]):int(prediction[3][1]),
                           int(prediction[0][0]):int(prediction[1][0]),
                           ]
        font = self.predict_font(first_block_part)
        color = self.get_mean_color(first_block_part)
        text_height = self.get_box_height(predictions[0][0])

        start_box = self._polygon_to_box(united_groups)
        b1 = start_box[2] * 0.02
        b2 = start_box[3] * 0.02
        print("bb: ", b1)
        start_box = (
            x + start_box[0] - b1,
            y + start_box[1] - b2,
            start_box[2] + b1 * 2,
            start_box[3] + b2 * 2,
        )
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(current_frame, start_box)

        current_frame_num = self.frame_num
        draw_text = True
        while current_frame_num <= self.frames_count:
            print("current_frame: ", current_frame_num)
            current_frame = self.frames[current_frame_num]
            success, box = tracker.update(current_frame)
            box = (box[0] - b1, box[1] - b2, box[2] + b1 * 2, box[3] + b2 * 2)

            if not success:
                break

            prediction = self.convert_box_to_polygon(box)

            current_frame = self.clear_text(current_frame, prediction)
            centroid = self.find_polygon_centroid(prediction)
            if centroid[0] < 0:
                draw_text = False
            if draw_text:
                current_frame = self.draw_text(
                    current_frame, new_text,
                    centroid[0], centroid[1],
                    text_height, color=color, font=font,
                )
            self.frames[current_frame_num] = current_frame

            current_frame_num += 1

        current_frame_num = self.frame_num - 1
        current_frame = self.frames[current_frame_num]
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(current_frame, start_box)
        draw_text = True

        while current_frame_num >= 0:
            print("current_frame: ", current_frame_num)
            current_frame = self.frames[current_frame_num]
            success, box = tracker.update(current_frame)
            box = (box[0] - b1, box[1] - b2, box[2] + b1 * 2, box[3] + b2 * 2)

            if not success:
                break

            prediction = self.convert_box_to_polygon(box)

            current_frame = self.clear_text(current_frame, prediction)
            centroid = self.find_polygon_centroid(prediction)
            if centroid[0] < 0:
                draw_text = False
            if draw_text:
                current_frame = self.draw_text(
                    current_frame, new_text,
                    centroid[0], centroid[1],
                    text_height, color=color, font=font,
                )
            self.frames[current_frame_num] = current_frame

            current_frame_num -= 1
        return True

    def translate_text(self, x, y, weight, height, src="en", dest="uk"):
        img_part = self.get_current_frame()[y:y + height, x:x + weight]
        prediction_groups, united_groups = self.find_text(img_part)
        if len(prediction_groups[0]) == 0:
            return None
        first_block_part = img_part[
                           int(united_groups[0][1]):int(united_groups[3][1]),
                           int(united_groups[0][0]):int(united_groups[1][0]),
                           ]
        text = pytesseract.image_to_string(first_block_part, lang='eng+ukr',
                                           config="--oem 1 --psm 6").strip()
        text = self.translate_deepl(text, source_lang=src, target_lang=dest)
        return text

    def remove_text(self, x, y, weight, height):
        current_frame = self.get_current_frame()

        img_part = current_frame[y:y + height, x:x + weight]
        predictions, united_groups = self.find_text(img_part)
        if len(predictions[0]) == 0:
            return False

        box = self._polygon_to_box(united_groups)
        box = (
            x + box[0],
            y + box[1],
            box[2],
            box[3],
        )
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(current_frame, box)

        def draw_boxes(frame, bbox):
            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

        current_frame_num = self.frame_num
        while current_frame_num <= self.frames_count:
            current_frame = self.frames[current_frame_num]
            success, box = tracker.update(current_frame)

            if not success:
                break

            prediction = self.convert_box_to_polygon(box)
            draw_boxes(current_frame, box)

            current_frame = self.clear_text(current_frame, prediction)
            self.frames[current_frame_num] = current_frame

            current_frame_num += 1
        return True

    def remove_text_revert(self, x, y, weight, height):
        current_frame = self.frames[self.frame_num - 1]

        img_part = current_frame[y:y + height, x:x + weight]
        predictions, united_groups = self.find_text(img_part)
        if len(predictions[0]) == 0:
            return False

        box = self._polygon_to_box(united_groups)
        box = (
            x + box[0],
            y + box[1],
            box[2],
            box[3],
        )
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(current_frame, box)

        current_frame_num = self.frame_num - 1
        while current_frame_num >= 0:
            current_frame = self.frames[current_frame_num]
            success, box = tracker.update(current_frame)

            if not success:
                break

            prediction = self.convert_box_to_polygon(box)

            current_frame = self.clear_text(current_frame, prediction)
            self.frames[current_frame_num] = current_frame

            current_frame_num -= 1
        return True

    def get_frames_count(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    def get_current_frame_num(self):
        return self.frame_num

    def set_frame_number(self, frame_number):
        self.frame_num = frame_number

    def next_frame(self):
        try:
            frame = self.frames[self.frame_num]
            ret = True
            self.frame_num += 1
        except IndexError:
            ret = False
            frame = self.frames[-1]
        return ret, frame

    def get_current_frame(self):
        return self.frames[self.frame_num]

    def get_time(self):
        current_time = self.frame_num / self.fps
        return f"{int(current_time // 3600):02}:{int((current_time % 3600) // 60):02}:{int(current_time % 60):02}"

    def convert_box_to_polygon(self, box):
        startX, startY, sizeX, sizeY = box
        endX = startX + sizeX
        endY = startY + sizeY

        top_left = (startX, startY)
        top_right = (endX, startY)
        bottom_right = (endX, endY)
        bottom_left = (startX, endY)

        polygon = np.array([top_left, top_right, bottom_right, bottom_left])

        return polygon

    def copy_text(self, x, y, weight, height):
        frame = self.frames[self.frame_num]
        img_part = frame[y:y + height, x:x + weight]
        text = pytesseract.image_to_string(img_part, lang='eng+ukr',
                                           config="--oem 1 --psm 6").strip()
        pyperclip.copy(text)

    def save_video(self, file_path):
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 20.0, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()

    def find_frame_with_text(self, text, step):
        text = text.lower()
        current_frame_num = 0
        while current_frame_num < self.frames_count:
            current_frame = self.frames[current_frame_num]
            founded_text = pytesseract.image_to_string(current_frame, lang='eng+ukr',
                                                       config="--oem 1 --psm 6").strip()
            if text in founded_text.lower():
                return current_frame_num
            current_frame_num += step

    def get_polygon_height(self, polygon):
        return int((polygon[3][1] - polygon[0][1]) * 1)
