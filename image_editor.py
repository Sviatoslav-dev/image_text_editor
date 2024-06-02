import sys

from PySide2.QtWidgets import *

from base_editor.editor_window import EditorWindow
from image_scene import ImageScene


class ImageEditor(EditorWindow):
    def __init__(self):
        super().__init__()
        pixmap_item = QGraphicsPixmapItem()
        self.scene = ImageScene(pixmap_item=pixmap_item)
        self.scene.addItem(pixmap_item)
        self.scene.rect = self.scene.addRect(0, 0, 0, 0)
        self.gv.setScene(self.scene)

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
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.close_file_action)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    app.exec_()
