import sys

import cv2
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QGraphicsScene, QGraphicsView

from drafts.design import Ui_Form


class PhotoScene(QGraphicsScene):
    def __init__(self, parent=None):
        super(PhotoScene, self).__init__(parent)

        self.rect = self.addRect(10, 10, 200, 200)
        # self.q_pixmap: QPixmap = q_pixmap
        # self.image = image

    def mousePressEvent(self, event):
        print(event.scenePos().x(), event.scenePos().y())
        self.rect.setRect(event.scenePos().x(), event.scenePos().y(), 0, 0)
        super(PhotoScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.rect.setRect(
            self.rect.rect().x(),
            self.rect.rect().y(),
            event.scenePos().x() - self.rect.rect().x(),
            event.scenePos().y() - self.rect.rect().y()
        )

    # def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
    #     print(self.rect.rect().width(), self.rect.rect().height())
    #     try:
    #         text = ""
    #         dialog = QDialog(self.parent())
    #         dialog_layout = QVBoxLayout()
    #         top_line_edit = QLineEdit(parent=dialog)
    #         dialog_layout.addWidget(top_line_edit)
    #
    #         def on_enter():
    #             nonlocal text
    #             text = top_line_edit.text()
    #             dialog.done(1)
    #
    #         top_line_edit.editingFinished.connect(on_enter)
    #         dialog.setLayout(dialog_layout)
    #         dialog.exec_()
    #
    #         numpy_image = self.image.img[
    #                       int(self.rect.rect().y()):int(
    #                           self.rect.rect().y() + self.rect.rect().height()),
    #                       int(self.rect.rect().x()):int(
    #                           self.rect.rect().x() + self.rect.rect().width())
    #                       ]
    #         predictions = find_text(numpy_image)
    #         numpy_image = clear_text(numpy_image, predictions[0])
    #         numpy_image = draw_text(
    #             numpy_image, text,
    #             predictions[0][0][1][3][0], predictions[0][0][1][3][1],
    #             get_box_height(predictions[0][0]),
    #         )
    #         self.image.img[
    #         int(self.rect.rect().y()):int(self.rect.rect().y() + self.rect.rect().height()),
    #         int(self.rect.rect().x()):int(self.rect.rect().x() + self.rect.rect().width())
    #         ] = numpy_image
    #         print(self.items())
    #         self.q_pixmap = QPixmap.fromImage(
    #             qimage2ndarray.array2qimage(cv2.cvtColor(self.image.img, cv2.COLOR_BGR2RGB)))
    #         self.clear()
    #         self.addPixmap(self.q_pixmap)
    #         self.rect = self.addRect(0, 0, 0, 0)
    #     except Exception as err:
    #         print(err)
    #         print(traceback.format_exception(*sys.exc_info()))
    #         raise err


class ThreadOpenCV(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, source):
        super().__init__()
        self.source = source

    def run(self):
        cap = cv2.VideoCapture(self.source)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                # p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(convertToQtFormat)

                cv2.waitKey(28)


class Widget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.label_video = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.label_video)

        self.ui.frame.setLayout(layout)
        self.scene = PhotoScene()
        self.scene.rect = self.scene.addRect(0, 0, 0, 0)
        self.graphics_view = QGraphicsView(self.scene)
        self.graphics_view.setStyleSheet("background:transparent;")
        layout.addWidget(self.graphics_view)
        # self.graphics_view.move(QPoint(0, 0))


        self.thread = ThreadOpenCV("video_with_text2.mp4")
        self.thread.changePixmap.connect(self.setImage)

        self.ui.pushButton.clicked.connect(self.playVideo)
    def playVideo(self):
        self.thread.start()
        self.graphics_view.move(QPoint(0, 0))

    def setImage(self, image):
        self.label_video.setPixmap(QPixmap.fromImage(image))


app = QtWidgets.QApplication(sys.argv)

mw = Widget()
mw.show()

sys.exit(app.exec_())
