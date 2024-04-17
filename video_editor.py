import sys

import cv2

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QVBoxLayout, QLineEdit

from drafts.clear_text import clear_text
from drafts.draw_text import draw_text
from east_text_detection import find_text_by_east
import qimage2ndarray


class VideoScene(QtWidgets.QGraphicsScene):
    text = ""

    def __init__(self, parent=None):
        super(VideoScene, self).__init__(parent)

        self.rect = self.addRect(0, 0, 0, 0)
        self.pause = False

    def mousePressEvent(self, event):
        if self.pause:
            print(event.scenePos().x(), event.scenePos().y())
            self.rect.setRect(event.scenePos().x(), event.scenePos().y(), 0, 0)
        super(VideoScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        if self.pause:
            self.rect.setRect(
                self.rect.rect().x(),
                self.rect.rect().y(),
                event.scenePos().x() - self.rect.rect().x(),
                event.scenePos().y() - self.rect.rect().y()
            )

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        dialog = QDialog(self.parent())
        dialog_layout = QVBoxLayout()
        top_line_edit = QLineEdit(parent=dialog)
        dialog_layout.addWidget(top_line_edit)

        def on_enter():
            self.text = top_line_edit.text()
            dialog.done(1)

        top_line_edit.editingFinished.connect(on_enter)
        dialog.setLayout(dialog_layout)
        dialog.exec_()


class VideoPlayer(QtWidgets.QWidget):
    pause = False
    first_stop = False

    def __init__(self, width=640, height=480, fps=30):
        QtWidgets.QWidget.__init__(self)

        self.video_size = QtCore.QSize(width, height)
        self.video_capture = cv2.VideoCapture()
        self.frame_timer = QtCore.QTimer()

        self.slider = QtWidgets.QSlider(Qt.Horizontal)

        self.setup_camera(fps)
        self.fps = fps

        graphic_view = QtWidgets.QGraphicsView()
        graphic_view.setFixedSize(self.video_size)
        self.scene = VideoScene()
        self.frame_label = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.frame_label)
        graphic_view.setScene(self.scene)
        self.scene.rect = self.scene.addRect(0, 0, 0, 0)

        self.quit_button = QtWidgets.QPushButton("Quit")
        self.play_pause_button = QtWidgets.QPushButton("Pause")
        self.camera_video_button = QtWidgets.QPushButton("Switch to video")

        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(graphic_view, 0, 0, 1, 2)
        self.main_layout.addWidget(self.slider, 3, 0)

        # self.slider.sliderPressed.connect(self.play_pause_button)
        self.slider.valueChanged.connect(self.seek_video)

        self.setup_ui()

        QtCore.QObject.connect(self.play_pause_button, QtCore.SIGNAL("clicked()"), self.play_pause)
        # self.tracker = None

    def setup_ui(self):
        self.quit_button.clicked.connect(self.close_win)

        self.main_layout.addWidget(self.play_pause_button, 1, 1, 1, 1)
        self.main_layout.addWidget(self.camera_video_button, 1, 0, 1, 1)
        self.main_layout.addWidget(self.quit_button, 2, 0, 1, 2)

        self.setLayout(self.main_layout)

    def play_pause(self):
        if not self.pause:
            self.frame_timer.stop()
            self.play_pause_button.setText("Play")
            # self.tracker = None
        else:
            self.frame_timer.start(int(1000 // self.fps))
            self.play_pause_button.setText("Pause")
            # self.video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.1)

        self.pause = not self.pause
        self.scene.pause = self.pause

    def seek_video(self):
        if self.video_capture:
            frame_number = self.slider.value()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            if self.pause and self.first_stop:
                ret, frame = self.video_capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = cv2.resize(frame, (self.video_size.width(), self.video_size.height()),
                                   interpolation=cv2.INTER_AREA)

                image = qimage2ndarray.array2qimage(frame)
                self.frame_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def setup_camera(self, fps):
        path = "/data/video_with_text2.mp4"
        self.video_capture.open(path)

        fps = self.video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print("duration", duration)

        self.slider.setMinimum(0)
        self.slider.setMaximum(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

        self.frame_timer.timeout.connect(self.display_video_stream)
        self.frame_timer.start(int(1000 // fps))

    def change_size(self, from_, to, x):
        return x / from_ * to

    def display_video_stream(self):
        ret, frame = self.video_capture.read()
        print(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
        print(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
        current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.setValue(current_frame)

        if not ret:
            return False

        if self.first_stop:
            height, width = frame.shape[:2]
            numpy_image = frame[
                          int(self.change_size(480, height, self.scene.rect.rect().y())):int(
                              self.change_size(480, height,
                                               self.scene.rect.rect().y() + self.scene.rect.rect().height())),
                          int(self.change_size(640, width, self.scene.rect.rect().x())):int(
                              self.change_size(640, width,
                                               self.scene.rect.rect().x() + self.scene.rect.rect().width()))
                          ]
            # if self.tracker is None:
            #     bbox = cv2.selectROI("Tracking", frame)
            #     self.tracker = cv2.legacy.TrackerCSRT_create()
            #     self.tracker.init(frame, bbox)
            # else:
            #     success, bbox = self.tracker.update(frame)

            # numpy_image = highlight_text(numpy_image)
            boxes = find_text_by_east(numpy_image)
            numpy_image = clear_text(numpy_image, box=(None, boxes[0]))

            # numpy_image = clear_text(numpy_image, box=bbox)
            numpy_image = draw_text(
                numpy_image, self.scene.text,
                boxes[0][3][0], boxes[0][3][1],
                boxes[0][3][1] - boxes[0][0][1],
            )
            frame[
            int(self.change_size(480, height, self.scene.rect.rect().y())):int(
                self.change_size(480, height,
                                 self.scene.rect.rect().y() + self.scene.rect.rect().height())),
            int(self.change_size(640, width, self.scene.rect.rect().x())):int(
                self.change_size(640, width,
                                 self.scene.rect.rect().x() + self.scene.rect.rect().width()))
            ] = numpy_image

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, (self.video_size.width(), self.video_size.height()),
                           interpolation=cv2.INTER_AREA)

        image = qimage2ndarray.array2qimage(frame)
        self.frame_label.setPixmap(QtGui.QPixmap.fromImage(image))

        if not self.first_stop:
            self.play_pause()
            self.first_stop = True

    def close_win(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
