import cv2
import numpy as np

from PySide2.QtGui import *
from PySide2.QtWidgets import *

from image_actions import ImageAction
from photo_editor import Main
from photo_model import PhotoModel
import qimage2ndarray

from translation_dialog import TranslationDialog


class CustomTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ShiftModifier:
                self.insertPlainText("\n")
            else:
                self.parent().accept()
        else:
            super().keyPressEvent(event)


class PhotoController:
    def __init__(self, window: Main, image_path, start_window):
        self.start_window = start_window
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
        self.window.translate_text.triggered.connect(self.set_action_translate)
        self.window.remove_action.triggered.connect(self.set_action_remove)
        self.window.copy_action.triggered.connect(self.set_action_copy)
        self.window.save_action.triggered.connect(self.save_image)
        self.window.close_file_action.triggered.connect(self.close_file)
        self.window.select_whole.triggered.connect(self.select_whole)
        self.window.undo.triggered.connect(self.undo)

    def set_action_replace(self):
        self.selected_action = ImageAction.ReplaceText
        self.replace_text()

    def set_action_remove(self):
        self.selected_action = ImageAction.RemoveText

    def set_action_copy(self):
        self.selected_action = ImageAction.CopyText

    def set_action_translate(self):
        dialog = TranslationDialog()
        dialog.exec_()
        self.image_model.translate_option = dialog.selected_option
        self.selected_action = ImageAction.TranslateText

    def update_image(self):
        {
            ImageAction.ReplaceText: self.replace_text,
            ImageAction.RemoveText: self.remove_text,
            ImageAction.CopyText: self.copy_text,
            ImageAction.TranslateText: self.translate,
        }[self.selected_action]()

    def replace_text(self):
        print(self.scene.rect.rect().width(), self.scene.rect.rect().height())
        text = ""
        dialog = QDialog(self.scene.parent())
        dialog_layout = QVBoxLayout()
        top_line_edit = CustomTextEdit(parent=dialog)
        dialog_layout.addWidget(top_line_edit)

        # def on_enter():
        #     nonlocal text
        #     text = top_line_edit.text()
        #     dialog.done(1)

        # top_line_edit.editingFinished.connect(on_enter)
        dialog.setLayout(dialog_layout)
        dialog.exec_()
        text = top_line_edit.toPlainText()

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

    def close_file(self):
        self.window.close()
        # start_window = Start()
        self.start_window.show()

    def select_whole(self):
        self.scene.rect.setRect(
            0,
            0,
            self.image_model.img.shape[1],
            self.image_model.img.shape[0]
        )
        self.update_image()

    def translate(self):
        print(self.scene.rect.rect().width(), self.scene.rect.rect().height())
        rect = self.scene.rect.rect()
        languges = self.image_model.translate_option
        text = self.image_model.translate_text(
            int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()), *languges
        )
        self.image_model.replace_text(
            text, int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

        print(self.scene.items())
        self.scene.q_pixmap.setPixmap(QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.image_model.img, cv2.COLOR_BGR2RGB))))

    def undo(self):
        self.image_model.img = np.copy(self.image_model.start_image)
        self.scene.q_pixmap.setPixmap(QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.image_model.img, cv2.COLOR_BGR2RGB))))
