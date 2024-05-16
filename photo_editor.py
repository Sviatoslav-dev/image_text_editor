import sys

from PySide2.QtCore import QSize, QCoreApplication
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from editor_window import EditorWindow
from photo_controller import PhotoController
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
