import cv2
import qimage2ndarray
from PySide2 import QtGui, QtCore

from video_model import VideoModel


class VideoController:
    pause = False
    first_stop = False

    def __init__(self, window):
        self.window = window
        self.scene = window.scene

        self.video_model = VideoModel()
        self.setup_camera(self.video_model.fps)
        self.window.quit_button.clicked.connect(self.close_win)
        self.window.slider.valueChanged.connect(self.seek_video)
        QtCore.QObject.connect(
            self.window.play_pause_button, QtCore.SIGNAL("clicked()"), self.play_pause)

        # self.window.show()

    def setup_camera(self, fps):
        self.window.slider.setMinimum(0)
        self.window.slider.setMaximum(self.video_model.get_frames_count())

        self.window.frame_timer.timeout.connect(lambda: self.display_video_stream())
        self.window.frame_timer.start(int(1000 // fps))

    def display_video_stream(self):
        ret, frame = self.video_model.video_capture.read()
        print(self.video_model.video_capture.get(cv2.CAP_PROP_POS_MSEC))
        print(self.video_model.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
        current_frame = self.video_model.get_current_frame()
        self.window.slider.setValue(current_frame)

        print(ret)
        if not ret:
            return False

        if self.first_stop:
            height, width = frame.shape[:2]
            x = int(self.change_size(640, width, self.scene.rect.rect().x()))
            y = int(self.change_size(480, height, self.scene.rect.rect().y()))
            rect_weight = int(self.change_size(640, width, self.scene.rect.rect().width()))
            rect_height = int(self.change_size(480, height, self.scene.rect.rect().height()))
            frame = self.video_model.replace_text(
                self.scene.text, x, y, rect_weight, rect_height, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, (self.window.video_size.width(), self.window.video_size.height()),
                           interpolation=cv2.INTER_AREA)

        image = qimage2ndarray.array2qimage(frame)
        self.window.frame_label.setPixmap(QtGui.QPixmap.fromImage(image))

        if not self.first_stop:
            self.play_pause()
            self.first_stop = True

    def play_pause(self):
        if not self.pause:
            self.window.frame_timer.stop()
            self.window.play_pause_button.setText("Play")
            # self.tracker = None
        else:
            self.window.frame_timer.start(int(1000 // self.video_model.fps))
            self.window.play_pause_button.setText("Pause")
            # self.video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.1)

        self.pause = not self.pause
        self.scene.pause = self.pause

    def change_size(self, from_, to, x):
        return x / from_ * to

    def seek_video(self):
        if self.video_model.video_capture:
            frame_number = self.window.slider.value()
            self.video_model.set_frame_number(frame_number)
            if self.pause and self.first_stop:
                ret, frame = self.video_model.video_capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = cv2.resize(frame, (self.window.video_size.width(), self.window.video_size.height()),
                                   interpolation=cv2.INTER_AREA)

                image = qimage2ndarray.array2qimage(frame)
                self.window.frame_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def close_win(self):
        self.video_model.video_capture.release()
        cv2.destroyAllWindows()
        self.window.close()

