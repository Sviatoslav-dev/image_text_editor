import sys

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QAction

from editor_window import EditorWindow
from video_controller import VideoController
from video_scene import VideoScene


class VideoPlayer(EditorWindow):

    def __init__(self, width=640, height=480, fps=30):
        super().__init__()

        self.video_size = QtCore.QSize(width, height)
        self.frame_timer = QtCore.QTimer()

        self.gv.setFixedSize(self.video_size)
        self.scene = VideoScene()
        self.frame_label = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.frame_label)
        self.gv.setScene(self.scene)
        self.scene.rect = self.scene.addRect(0, 0, 0, 0)

        # self.quit_button = QtWidgets.QPushButton("Quit")
        self.play_pause_button = QtWidgets.QPushButton("Pause")
        # self.camera_video_button = QtWidgets.QPushButton("Switch to video")

        self.video_layout = QtWidgets.QGridLayout()
        # self.video_layout.addWidget(self.gv, 0, 0, 1, 2)
        # self.video_layout.addWidget(self.slider, 3, 0)

        self.setup_ui()

        self.replace_action = QAction("Replace text", self)
        self.replace_action.setCheckable(True)
        self.replace_action.setChecked(True)
        self.remove_action = QAction("Remove text", self)
        self.remove_action.setCheckable(True)
        self.copy_action = QAction("Copy text", self)
        self.copy_action.setCheckable(True)
        self.find_action = QAction("Find text", self)
        self.save_action = QAction("Save", self)

        self.tools_group.addAction(self.replace_action)
        self.tools_group.addAction(self.remove_action)
        self.tools_group.addAction(self.copy_action)

        self.toolbar.addAction(self.replace_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.remove_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.copy_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.find_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.save_action)

    def setup_ui(self):
        self.video_layout.addWidget(self.play_pause_button, 1, 1, 1, 1)
        # self.video_layout.addWidget(self.camera_video_button, 1, 0, 1, 1)
        # self.main_layout.addWidget(self.quit_button, 2, 0, 1, 2)

        # self.setLayout(self.video_layout)
        self.vbox1.addLayout(self.video_layout)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    # player.show()
    VideoController(player, "data/wideo_with_text.mp4")
    player.show()
    sys.exit(app.exec_())
