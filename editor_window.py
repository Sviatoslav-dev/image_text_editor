import sys

import PySide2
from PySide2 import QtCore
from PySide2.QtCore import QSize, QCoreApplication
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class EditorWindow(QWidget):
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
        self.verticalLayout_4.addLayout(self.vbox3)
        self.vbox.addWidget(self.base_frame)
        self.hbox.addLayout(self.vbox)
        self.verticalLayout_2.addLayout(self.hbox)

        self.retranslateUi()

        self.setWindowIcon(
            QIcon("C:/Users/slavi/PycharmProjects/image_text_editor/data/icon/icon.png"))
        self.move(120, 100)


        self.toolbar = QToolBar()
        self.toolbar.setOrientation(PySide2.QtCore.Qt.Orientation.Vertical)
        self.toolbar.setStyleSheet("""
                   QToolBar {
                       spacing: 10px;
                   }
                   QToolButton {
                       font-size: 18px;
                       padding: 10px;
                       min-width: 80px;
                       min-height: 40px;
                   }
               """)
        self.vbox3.addWidget(self.toolbar)

        self.tools_group = QActionGroup(self)
        self.tools_group.setExclusive(True)

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("Editor", "Editor"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = Start()
    # window = Main(["C:/Users/slavi/PycharmProjects/image_text_editor/ui/landscape.jpg"])
    # window = Main(["C:/Users/slavi/PycharmProjects/image_text_editor/ui/landscape.jpg"])
    window = EditorWindow()
    window.show()
    app.exec_()
