import sys

from PySide2.QtWidgets import *

from editor_window import EditorWindow
from photo_scene import PhotoScene


class Main(EditorWindow):
    def __init__(self):
        # initialize
        super().__init__()
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

        self.replace_action = QAction("Replace text", self)
        self.replace_action.setCheckable(True)
        self.replace_action.setChecked(True)
        self.remove_action = QAction("Remove text", self)
        self.remove_action.setCheckable(True)
        self.copy_action = QAction("Copy text", self)
        self.copy_action.setCheckable(True)
        self.select_whole = QAction("Select whole", self)
        self.save_action = QAction("Save", self)
        self.close_file_action = QAction("Close file", self)

        self.tools_group.addAction(self.replace_action)
        self.tools_group.addAction(self.remove_action)
        self.tools_group.addAction(self.copy_action)

        self.toolbar.addAction(self.replace_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.remove_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.copy_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.select_whole)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.close_file_action)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = Start()
    # window = Main(["C:/Users/slavi/PycharmProjects/image_text_editor/ui/landscape.jpg"])
    # window = Main(["C:/Users/slavi/PycharmProjects/image_text_editor/ui/landscape.jpg"])
    window = Main()
    window.show()
    app.exec_()
