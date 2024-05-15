import cv2

from base_model import BaseImageModel
from east_text_detection import find_text_by_east


class VideoModel(BaseImageModel):
    def __init__(self):
        super().__init__()
        self.video_capture = cv2.VideoCapture()

        fps = 30
        self.setup_camera(fps)
        self.fps = fps

    def setup_camera(self, fps):
        path = "data/video_with_text2.mp4"
        self.video_capture.open(path)

        fps = self.video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print("duration", duration)

    def replace_text(self, new_text, x, y, weight, height, frame):
        img_part = frame[y:y + height, x:x + weight]
        boxes = find_text_by_east(img_part)
        img_part = self.clear_text(img_part, box=(None, boxes[0]))
        img_part = self.draw_text(
            img_part, new_text,
            boxes[0][3][0], boxes[0][3][1],
            boxes[0][3][1] - boxes[0][0][1],
        )
        frame[y:y + height, x:x + weight] = img_part
        return frame

    def get_frames_count(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    def get_current_frame(self):
        return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))

    def set_frame_number(self, frame_number):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
