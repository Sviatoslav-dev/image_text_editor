import sys

import cv2

from PySide2.QtCore import QSize, QCoreApplication
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from photo_controller import PhotoController
from photo_scene import PhotoScene
from ui.scripts import Images
import qimage2ndarray


class Main(QWidget):
    def __init__(self):
        # initialize
        super().__init__()
        self.setObjectName("Editor")
        self.resize(1345, 859)
        self.verticalLayout_2 = QVBoxLayout(self)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.hbox = QHBoxLayout()
        self.hbox.setSpacing(0)
        self.hbox.setObjectName("hbox")
        self.vbox1 = QVBoxLayout()
        self.vbox1.setSpacing(20)
        self.vbox1.setObjectName("vbox1")
        self.gv = QGraphicsView(self)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gv.sizePolicy().hasHeightForWidth())
        self.gv.setSizePolicy(sizePolicy)
        self.gv.setMaximumSize(QSize(1200, 900))
        self.gv.setStyleSheet("QGraphicsView{border: 0px}")
        self.gv.setFrameShadow(QFrame.Sunken)
        self.gv.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gv.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gv.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        brush = QBrush(QColor(240, 240, 240))
        brush.setStyle(Qt.SolidPattern)
        self.gv.setBackgroundBrush(brush)
        self.gv.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
        self.gv.setObjectName("gv")
        self.vbox1.addWidget(self.gv)
        self.slider = QSlider(self)
        self.slider.setMaximum(360)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setObjectName("slider")
        self.vbox1.addWidget(self.slider)
        self.hbox5 = QHBoxLayout()
        self.hbox5.setContentsMargins(200, -1, 200, -1)
        self.hbox5.setSpacing(100)
        self.hbox5.setObjectName("hbox5")
        self.pushButton = QPushButton(self)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.hbox5.addWidget(self.pushButton)
        spacerItem = QSpacerItem(140, 20, QSizePolicy.Fixed,
                                 QSizePolicy.Minimum)
        self.hbox5.addItem(spacerItem)
        self.pushButton_2 = QPushButton(self)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.hbox5.addWidget(self.pushButton_2)
        self.vbox1.addLayout(self.hbox5)
        self.hbox.addLayout(self.vbox1)
        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(0, -1, 0, -1)
        self.vbox.setSpacing(0)
        self.vbox.setObjectName("vbox")
        self.base_frame = QFrame(self)
        self.base_frame.setEnabled(True)
        self.base_frame.setFrameShape(QFrame.StyledPanel)
        self.base_frame.setFrameShadow(QFrame.Raised)
        self.base_frame.setObjectName("base_frame")
        self.verticalLayout_4 = QVBoxLayout(self.base_frame)
        self.verticalLayout_4.setContentsMargins(10, -1, 10, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.vbox3 = QVBoxLayout()
        self.vbox3.setSpacing(120)
        self.vbox3.setObjectName("vbox3")
        self.filter_btn = QPushButton(self.base_frame)
        sizePolicy = QSizePolicy(QSizePolicy.Maximum,
                                 QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filter_btn.sizePolicy().hasHeightForWidth())
        self.filter_btn.setSizePolicy(sizePolicy)
        self.filter_btn.setMinimumSize(QSize(0, 0))
        self.filter_btn.setMaximumSize(QSize(800, 60))
        font = QFont()
        font.setFamily("Neue Haas Grotesk Text Pro")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.filter_btn.setFont(font)
        self.filter_btn.setObjectName("filter_btn")
        self.vbox3.addWidget(self.filter_btn)
        self.adjust_btn = QPushButton(self.base_frame)
        self.adjust_btn.setMaximumSize(QSize(16777215, 60))
        font = QFont()
        font.setFamily("Neue Haas Grotesk Text Pro")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.adjust_btn.setFont(font)
        self.adjust_btn.setObjectName("adjust_btn")
        self.vbox3.addWidget(self.adjust_btn)
        self.ai_btn = QPushButton(self.base_frame)
        self.ai_btn.setMaximumSize(QSize(16777215, 60))
        font = QFont()
        font.setFamily("Neue Haas Grotesk Text Pro")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ai_btn.setFont(font)
        self.ai_btn.setObjectName("ai_btn")
        self.vbox3.addWidget(self.ai_btn)
        spacerItem1 = QSpacerItem(0, 100, QSizePolicy.Minimum,
                                  QSizePolicy.Fixed)
        self.vbox3.addItem(spacerItem1)
        self.save_btn = QPushButton(self.base_frame)
        self.save_btn.setMaximumSize(QSize(16777215, 60))
        font = QFont()
        font.setFamily("Neue Haas Grotesk Text Pro")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.save_btn.setFont(font)
        self.save_btn.setObjectName("save_btn")
        self.vbox3.addWidget(self.save_btn)
        self.verticalLayout_4.addLayout(self.vbox3)
        self.vbox.addWidget(self.base_frame)
        self.hbox.addLayout(self.vbox)
        self.verticalLayout_2.addLayout(self.hbox)

        self.retranslateUi()
        # QMetaObject.connectSlotsByName(self)
        # loadUi(f"{pathlib.Path(__file__).parent.absolute()}\\ui\\main.ui", self)
        self.setWindowIcon(
            QIcon("C:/Users/slavi/PycharmProjects/image_text_editor/data/icon/icon.png"))
        self.move(120, 100)

        # self.img = QPixmap(
        #     qimage2ndarray.array2qimage(cv2.cvtColor(self.img_class.img, cv2.COLOR_BGR2RGB)))

        # self.gv = self.findChild(QGraphicsView, "gv")
        # self.scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem()
        self.scene = PhotoScene(pixmap_item=pixmap_item)
        # self.scene.mouseReleaseEvent
        self.scene.addItem(pixmap_item)
        # pixmap_item.setPixmap(self.img)
        # self.scene_img = self.scene.addPixmap(self.img)
        self.scene.rect = self.scene.addRect(0, 0, 0, 0)
        self.gv.setScene(self.scene)

        self.zoom_moment = False
        self._zoom = 0

        self.rotate_value, self.brightness_value, self.contrast_value, self.saturation_value = 0, 0, 1, 0
        self.flip = [False, False]
        self.zoom_factor = 1

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("Editor", "Editor"))
        self.pushButton.setText(_translate("Editor", "<<"))
        self.pushButton_2.setText(_translate("Editor", ">>"))
        self.filter_btn.setText(_translate("Editor", "          Filter          "))
        self.adjust_btn.setText(_translate("Editor", "          Adjust          "))
        self.ai_btn.setText(_translate("Editor", "AI"))
        self.save_btn.setText(_translate("Editor", "Save"))

    def get_zoom_factor(self):
        return self.zoom_factor

    def click_save(self):
        try:
            file, _ = QFileDialog.getSaveFileName(self, 'Save File', f"{self.img_class.img_name}."
                                                                     f"{self.img_class.img_format}",
                                                  "Image Files (*.jpg *.png *.jpeg *.ico);;All Files (*)")
            self.img_class.save_img(file)
        except Exception:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = Start()
    # window = Main(["C:/Users/slavi/PycharmProjects/image_text_editor/ui/landscape.jpg"])
    # window = Main(["C:/Users/slavi/PycharmProjects/image_text_editor/ui/landscape.jpg"])
    window = Main()
    # window.show()
    controller = PhotoController(
        window,
        "C:/Users/slavi/PycharmProjects/image_text_editor/data/img_5.png",
    )
    app.exec_()
