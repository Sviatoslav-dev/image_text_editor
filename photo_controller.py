import cv2

from PySide2.QtGui import *
from PySide2.QtWidgets import *

from image_actions import ImageAction
from photo_editor import Main
from photo_model import PhotoModel
import qimage2ndarray


class PhotoController:
    def __init__(self, window: Main, image_path):
        self.selected_action = ImageAction.ReplaceText
        self.window = window
        self.scene = window.scene
        self.scene.update_image = self.update_image
        self.image_model = PhotoModel()

        self.image_model.read_image(image_path)
        self.scene.q_pixmap.setPixmap((QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.image_model.img, cv2.COLOR_BGR2RGB)))))
        window.show()

        self.window.replace_action.triggered.connect(self.set_action_replace)
        self.window.remove_action.triggered.connect(self.set_action_remove)
        self.window.copy_action.triggered.connect(self.set_action_copy)
        self.window.save_action.triggered.connect(self.save_image)

    def set_action_replace(self):
        self.selected_action = ImageAction.ReplaceText

    def set_action_remove(self):
        self.selected_action = ImageAction.RemoveText

    def set_action_copy(self):
        self.selected_action = ImageAction.CopyText

    def update_image(self):
        {
            ImageAction.ReplaceText: self.replace_text,
            ImageAction.RemoveText: self.remove_text,
            ImageAction.CopyText: self.copy_text,
        }[self.selected_action]()

    def replace_text(self):
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

    def remove_text(self):
        print(self.scene.rect.rect().width(), self.scene.rect.rect().height())
        rect = self.scene.rect.rect()
        self.image_model.remove_text(
            int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

        print(self.scene.items())
        self.scene.q_pixmap.setPixmap(QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.image_model.img, cv2.COLOR_BGR2RGB))))

    def copy_text(self):
        print(self.scene.rect.rect().width(), self.scene.rect.rect().height())
        rect = self.scene.rect.rect()
        self.image_model.copy_text(
            int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

    def save_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self.window, "Save Image As", "", "PNG Files (*.png);;All Files (*)", options=options)

        if file_path:
            self.window.scene.q_pixmap.pixmap().save(file_path)
            print(f"Image saved to {file_path}")
