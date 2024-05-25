import sys

from PySide2.QtWidgets import *

from image_controller import ImageController
from image_editor import ImageEditor

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = Start()
    # window = Main(["C:/Users/slavi/PycharmProjects/image_text_editor/ui/landscape.jpg"])
    # window = Main(["C:/Users/slavi/PycharmProjects/image_text_editor/ui/landscape.jpg"])
    window = ImageEditor()
    # window.show()
    controller = ImageController(
        window,
        "data/img_11.png",
    )
    app.exec_()
