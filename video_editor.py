import sys

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QAction, QSlider, QLabel, QHBoxLayout, QVBoxLayout

from editor_window import EditorWindow
from video_controller import VideoController
from video_scene import VideoScene


class VideoPlayer(EditorWindow):

    def __init__(self, width=1280, height=800, fps=30):
        super().__init__()
        self.slider = QSlider(self)
        self.slider.setMaximum(360)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setObjectName("slider")
        self.vbox1.addStretch()
        self.vbox1.addWidget(self.slider)

        self.video_size = QtCore.QSize(width, height)
        self.video_size.width()
        self.frame_timer = QtCore.QTimer()

        self.gv.setFixedSize(self.video_size)
        self.scene = VideoScene()
        self.frame_label = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.frame_label)
        self.gv.setScene(self.scene)
        self.scene.rect = self.scene.addRect(0, 0, 0, 0)

        # self.quit_button = QtWidgets.QPushButton("Quit")
        self.time_code_label = QLabel("00:00:00", self)
        self.play_pause_button = QtWidgets.QPushButton("Pause")
        # self.camera_video_button = QtWidgets.QPushButton("Switch to video")

        self.video_layout = QtWidgets.QGridLayout()
        # self.video_layout.addWidget(self.gv, 0, 0, 1, 2)
        # self.video_layout.addWidget(self.slider, 3, 0)

        self.setup_ui()

        self.replace_action = QAction("Replace text", self)
        self.replace_action.setCheckable(True)
        self.replace_action.setChecked(True)
        self.translate_text = QAction("Translate", self)
        self.translate_text.setCheckable(True)
        self.remove_action = QAction("Remove text", self)
        self.remove_action.setCheckable(True)
        self.copy_action = QAction("Copy text", self)
        self.copy_action.setCheckable(True)
        self.select_whole = QAction("Select whole", self)
        self.undo = QAction("Undo", self)
        self.find_action = QAction("Find text", self)
        self.save_action = QAction("Save", self)
        self.close_file_action = QAction("Close file", self)

        self.tools_group.addAction(self.replace_action)
        self.tools_group.addAction(self.translate_text)
        self.tools_group.addAction(self.remove_action)
        self.tools_group.addAction(self.copy_action)

        self.toolbar.addAction(self.replace_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.translate_text)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.remove_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.copy_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.select_whole)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.undo)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.find_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.close_file_action)

    def setup_ui(self):
        # self.video_layout.addWidget(self.time_code_label, 0, 2, 1, 1, Qt.AlignRight)
        # self.video_layout.addWidget(self.play_pause_button, 0, 1, 1, 1, Qt.AlignCenter)
        # # self.video_layout.addWidget(self.camera_video_button, 1, 0, 1, 1)
        # # self.main_layout.addWidget(self.quit_button, 2, 0, 1, 2)
        #
        # # self.setLayout(self.video_layout)
        # self.vbox1.addLayout(self.video_layout)

        self.control_h_layout = QHBoxLayout()

        self.control_v_layout = QVBoxLayout()
        self.control_v_layout.addWidget(self.time_code_label, alignment=Qt.AlignCenter)
        self.control_v_layout.addWidget(self.play_pause_button)
        self.control_v_layout.addStretch()

        self.control_h_layout.addStretch()
        self.control_h_layout.addLayout(self.control_v_layout)
        self.control_h_layout.addStretch()

        self.vbox1.addLayout(self.control_h_layout)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    # player.show()
    VideoController(player, "data/video_with_text8.mp4")
    player.show()
    sys.exit(app.exec_())
