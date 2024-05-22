import cv2
import numpy as np
import pyperclip

from base_model import BaseImageModel
from east_text_detection import find_text_by_east


class VideoModel(BaseImageModel):
    def __init__(self, video_path):
        super().__init__()
        self.video_capture = cv2.VideoCapture()

        fps = 30
        self.setup_camera(fps, video_path)
        self.fps = fps
        self.frame_num = -1
        self.frames = []
        self.frames_count = self.get_frames_count()
        self.read_all_frames()

    def read_all_frames(self):
        self.frames = []
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            self.frames.append(frame)

    def setup_camera(self, fps, path):
        self.video_capture.open(path)

        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print("duration", duration)

    def replace_text(self, new_text, x, y, weight, height):
        current_frame = self.get_current_frame()

        img_part = current_frame[y:y + height, x:x + weight]
        predictions, united_groups = self.find_text(img_part)
        prediction = predictions[0][0][1]
        first_block_part = img_part[
                           int(prediction[0][1]):int(prediction[3][1]),
                           int(prediction[0][0]):int(prediction[1][0]),
                           ]
        font = self.predict_font(first_block_part)
        color = self.get_mean_color(first_block_part)
        text_height = self.get_box_height(predictions[0][0])

        box = self._polygon_to_box(united_groups)
        box = (
            x + box[0],
            y + box[1],
            box[2],
            box[3],
        )
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(current_frame, box)

        current_frame_num = self.frame_num
        while current_frame_num <= self.frames_count:
            print("current_frame_num: ", current_frame_num)
            current_frame = self.frames[current_frame_num]
            success, box = tracker.update(current_frame)

            if not success:
                break

            prediction = self.convert_box_to_polygon(box)

            current_frame = self.clear_text(current_frame, prediction)
            centroid = self.find_polygon_centroid(prediction)
            current_frame = self.draw_text(
                current_frame, new_text,
                centroid[0], centroid[1],
                text_height, color=color, font=font,
            )
            self.frames[current_frame_num] = current_frame

            current_frame_num += 1

    def replace_text_revert(self, new_text, x, y, weight, height):
        current_frame = self.frames[self.frame_num - 1]

        img_part = current_frame[y:y + height, x:x + weight]
        predictions, united_groups = self.find_text(img_part)
        prediction = predictions[0][0][1]
        first_block_part = img_part[
                           int(prediction[0][1]):int(prediction[3][1]),
                           int(prediction[0][0]):int(prediction[1][0]),
                           ]
        font = self.predict_font(first_block_part)
        color = self.get_mean_color(first_block_part)
        text_height = self.get_box_height(predictions[0][0])

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
            print("current_frame_num: ", current_frame_num)
            current_frame = self.frames[current_frame_num]
            success, box = tracker.update(current_frame)

            if not success:
                break

            prediction = self.convert_box_to_polygon(box)

            current_frame = self.clear_text(current_frame, prediction)
            centroid = self.find_polygon_centroid(prediction)
            current_frame = self.draw_text(
                current_frame, new_text,
                centroid[0], centroid[1],
                text_height, color=color, font=font,
            )
            self.frames[current_frame_num] = current_frame

            current_frame_num -= 1

    def remove_text(self, x, y, weight, height):
        current_frame = self.get_current_frame()

        img_part = current_frame[y:y + height, x:x + weight]
        predictions, united_groups = self.find_text(img_part)

        box = self._polygon_to_box(united_groups)
        box = (
            x + box[0],
            y + box[1],
            box[2],
            box[3],
        )
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(current_frame, box)

        current_frame_num = self.frame_num
        while current_frame_num <= self.frames_count:
            print("current_frame_num: ", current_frame_num)
            current_frame = self.frames[current_frame_num]
            success, box = tracker.update(current_frame)

            if not success:
                break

            prediction = self.convert_box_to_polygon(box)

            current_frame = self.clear_text(current_frame, prediction)
            self.frames[current_frame_num] = current_frame

            current_frame_num += 1

    def remove_text_revert(self, x, y, weight, height):
        current_frame = self.frames[self.frame_num - 1]

        img_part = current_frame[y:y + height, x:x + weight]
        predictions, united_groups = self.find_text(img_part)

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
            print("current_frame_num: ", current_frame_num)
            current_frame = self.frames[current_frame_num]
            success, box = tracker.update(current_frame)

            if not success:
                break

            prediction = self.convert_box_to_polygon(box)

            current_frame = self.clear_text(current_frame, prediction)
            self.frames[current_frame_num] = current_frame

            current_frame_num -= 1

    def get_frames_count(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    def get_current_frame_num(self):
        # return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        return self.frame_num

    def set_frame_number(self, frame_number):
        # self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.frame_num = frame_number

    def next_frame(self):
        # return self.video_capture.read()
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
        prediction_groups, united_groups = self.find_text(img_part)
        lines = self.group_text_by_lines(prediction_groups[0])

        text = '\n'.join(lines)
        pyperclip.copy(text)

    def save_video(self, file_path):
        # Налаштування VideoWriter
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 20.0, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()
        print(f"Video saved to {file_path}")

    def find_frame_with_text(self, text, step):
        print("frames: ", self.get_frames_count())
        text = text.lower()
        current_frame_num = 0
        while current_frame_num < self.frames_count:
            print("current_frame_num: ", current_frame_num)
            current_frame = self.frames[current_frame_num]
            predictions, united_groups = self.find_text(current_frame)
            founded_text = ""
            for prediction in predictions[0]:
                founded_text += " " + prediction[0]
            if text in founded_text:
                return current_frame_num
            current_frame_num += step

    def get_polygon_height(self, polygon):
        return int((polygon[3][1] - polygon[0][1]) * 1)
