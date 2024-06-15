import cv2
import numpy as np

from PySide2.QtGui import *
from PySide2.QtWidgets import *

from base_editor.base_controller import BaseController
from image_actions import ImageAction
from image_editor import ImageEditor
from image_model import ImageModel
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


class ImageController(BaseController):
    def __init__(self, window: ImageEditor, image_path, start_window):
        super().__init__(window, image_path, start_window)
        self.selected_action = ImageAction.ReplaceText
        self.scene = window.scene
        self.scene.update_image = self.update_image
        self.model = ImageModel()

        self.model.read_image(image_path)
        self.scene.q_pixmap.setPixmap((QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.model.img, cv2.COLOR_BGR2RGB)))))
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
        self.model.translate_option = dialog.selected_option
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
        dialog.setWindowTitle("New text")
        dialog_layout = QVBoxLayout()
        top_line_edit = CustomTextEdit(parent=dialog)
        dialog_layout.addWidget(top_line_edit)

        dialog.setLayout(dialog_layout)
        dialog.exec_()
        text = top_line_edit.toPlainText()

        rect = self.scene.rect.rect()
        success = self.model.replace_text(
            text, int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

        if not success:
            self.show_not_found_message()

        print(self.scene.items())
        self.scene.q_pixmap.setPixmap(QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.model.img, cv2.COLOR_BGR2RGB))))

    def remove_text(self):
        print(self.scene.rect.rect().width(), self.scene.rect.rect().height())
        rect = self.scene.rect.rect()
        success = self.model.remove_text(
            int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        if not success:
            self.show_not_found_message()

        print(self.scene.items())
        self.scene.q_pixmap.setPixmap(QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.model.img, cv2.COLOR_BGR2RGB))))

    def copy_text(self):
        print(self.scene.rect.rect().width(), self.scene.rect.rect().height())
        rect = self.scene.rect.rect()
        self.model.copy_text(
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
            self.model.img.shape[1],
            self.model.img.shape[0]
        )
        self.update_image()

    def translate(self):
        print(self.scene.rect.rect().width(), self.scene.rect.rect().height())
        rect = self.scene.rect.rect()
        languges = self.model.translate_option
        text = self.model.translate_text(
            int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()), *languges
        )
        if text is None:
            self.show_not_found_message()
        success = self.model.replace_text(
            text, int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

        if not success:
            self.show_not_found_message()

        print(self.scene.items())
        self.scene.q_pixmap.setPixmap(QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.model.img, cv2.COLOR_BGR2RGB))))

    def undo(self):
        self.model.img = np.copy(self.model.start_image)
        self.scene.q_pixmap.setPixmap(QPixmap.fromImage(
            qimage2ndarray.array2qimage(cv2.cvtColor(self.model.img, cv2.COLOR_BGR2RGB))))
