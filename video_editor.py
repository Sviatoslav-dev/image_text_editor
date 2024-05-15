import sys

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt

from video_controller import VideoController
from video_scene import VideoScene


class VideoPlayer(QtWidgets.QWidget):

    def __init__(self, width=640, height=480, fps=30):
        QtWidgets.QWidget.__init__(self)

        self.video_size = QtCore.QSize(width, height)
        self.frame_timer = QtCore.QTimer()

        self.slider = QtWidgets.QSlider(Qt.Horizontal)

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

        self.setup_ui()
        # self.tracker = None

    def setup_ui(self):
        self.main_layout.addWidget(self.play_pause_button, 1, 1, 1, 1)
        self.main_layout.addWidget(self.camera_video_button, 1, 0, 1, 1)
        self.main_layout.addWidget(self.quit_button, 2, 0, 1, 2)

        self.setLayout(self.main_layout)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    # player.show()
    VideoController(player)
    player.show()
    sys.exit(app.exec_())
