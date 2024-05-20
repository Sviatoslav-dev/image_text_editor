import sys

from PySide2.QtCore import QSize, QCoreApplication

from PySide2.QtGui import *
from PySide2.QtWidgets import *

from photo_controller import PhotoController
from photo_editor import Main
from video_controller import VideoController
from video_editor import VideoPlayer


class Start(QWidget):
    def __init__(self):
        super().__init__()
        # loadUi(f"{pathlib.Path(__file__).parent.absolute()}\\ui\\startup.ui", self)
        self.setObjectName("Editor")
        self.resize(720, 400)
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.NoFocus)
        self.verticalLayout_2 = QVBoxLayout(self)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.vbox = QVBoxLayout()
        self.vbox.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.vbox.setContentsMargins(-1, 0, -1, -1)
        self.vbox.setSpacing(0)
        self.vbox.setObjectName("vbox")
        self.vbox2 = QVBoxLayout()
        self.vbox2.setContentsMargins(0, 120, -1, -1)
        self.vbox2.setSpacing(0)
        self.vbox2.setObjectName("vbox2")
        self.title = QLabel(self)
        self.title.setLayoutDirection(Qt.LeftToRight)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setObjectName("title")
        self.vbox2.addWidget(self.title)
        self.vbox.addLayout(self.vbox2)
        self.hbox = QHBoxLayout()
        self.hbox.setContentsMargins(-1, -1, -1, 110)
        self.hbox.setSpacing(0)
        self.hbox.setObjectName("hbox")
        self.browse = QPushButton(self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browse.sizePolicy().hasHeightForWidth())
        self.browse.setSizePolicy(sizePolicy)
        self.browse.setMaximumSize(QSize(150, 70))
        self.browse.setSizeIncrement(QSize(0, 0))
        self.browse.setBaseSize(QSize(0, 0))
        font = QFont()
        font.setFamily("Neue Haas Grotesk Text Pro Medi")
        font.setPointSize(10)
        self.browse.setFont(font)
        self.browse.setFocusPolicy(Qt.NoFocus)
        self.browse.setLayoutDirection(Qt.LeftToRight)
        self.browse.setIconSize(QSize(19, 19))
        self.browse.setAutoDefault(False)
        self.browse.setDefault(False)
        self.browse.setFlat(False)
        self.browse.setObjectName("browse")
        self.hbox.addWidget(self.browse)
        self.vbox.addLayout(self.hbox)
        self.verticalLayout_2.addLayout(self.vbox)

        self.retranslateUi()
        # self.QMetaObject.connectSlotsByName(self)

        self.setWindowIcon(
            QIcon("C:/Users/slavi/PycharmProjects/image_text_editor/data/icon/icon.png"))

        self.button = self.findChild(QPushButton, "browse")
        self.button.clicked.connect(self.open_file)
        self.files, self.main_window = None, None

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("Editor", "Image Editor"))
        self.title.setText(_translate("Editor",
                                      "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#aa007f;\">Start from choosing your files</span></p></body></html>"))
        self.browse.setText(_translate("Editor", "Browse here"))

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                                   "All Files (*);;Image Files (*.jpg *.jpeg *.png *.bmp);;Video Files (*.avi *.mp4 *.mov *.mkv)",
                                                   options=options)

        if file_path:
            self.files = file_path
            print(self.files)
            self.close()
            if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                window = Main()
                # window.show()
                PhotoController(
                    window,
                    file_path,
                    # "data/img_7.png",
                )
            elif file_path.endswith(('.avi', '.mp4', '.mov', '.mkv')):
                player = VideoPlayer()
                # player.show()
                VideoController(
                    player,
                    file_path,
                    # "data/wideo_with_text.mp4",
                )
                player.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Start()
    window.show()
    app.exec_()
