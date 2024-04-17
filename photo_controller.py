import cv2

from PySide2.QtGui import *
from PySide2.QtWidgets import *
import qimage2ndarray

from photo_model import PhotoModel


class PhotoController:
    def __init__(self, window, image_path):
        self.window = window
        self.scene = window.scene
        self.scene.update_image = self.update_image
        self.image_model = PhotoModel()

        self.image_model.read_image(image_path)
        self.scene.q_pixmap.setPixmap((QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.image_model.img, cv2.COLOR_BGR2RGB)))))
        window.show()

    def update_image(self):
        print(self.scene.rect.rect().width(), self.scene.rect.rect().height())
        text = ""
        dialog = QDialog(self.scene.parent())
        dialog_layout = QVBoxLayout()
        top_line_edit = QLineEdit(parent=dialog)
        dialog_layout.addWidget(top_line_edit)

        def on_enter():
            nonlocal text
            text = top_line_edit.text()
            dialog.done(1)

        top_line_edit.editingFinished.connect(on_enter)
        dialog.setLayout(dialog_layout)
        dialog.exec_()

        rect = self.scene.rect.rect()
        self.image_model.replace_text(
            text, int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

        print(self.scene.items())
        self.scene.q_pixmap.setPixmap(QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.image_model.img, cv2.COLOR_BGR2RGB))))

