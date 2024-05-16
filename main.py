import sys

from PySide2.QtWidgets import *

from photo_controller import PhotoController
from photo_editor import Main

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
