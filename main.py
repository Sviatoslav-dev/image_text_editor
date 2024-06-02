import sys

from PySide2.QtWidgets import *
from start import Start

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Start()
    window.show()
    app.exec_()
