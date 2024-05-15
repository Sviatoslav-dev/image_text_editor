from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import QDialog, QVBoxLayout, QLineEdit

class VideoScene(QtWidgets.QGraphicsScene):
    text = ""

    def __init__(self, parent=None):
        super(VideoScene, self).__init__(parent)

        self.rect = self.addRect(0, 0, 0, 0)
        self.pause = False

    def mousePressEvent(self, event):
        if self.pause:
            print(event.scenePos().x(), event.scenePos().y())
            self.rect.setRect(event.scenePos().x(), event.scenePos().y(), 0, 0)
        super(VideoScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        if self.pause:
            self.rect.setRect(
                self.rect.rect().x(),
                self.rect.rect().y(),
                event.scenePos().x() - self.rect.rect().x(),
                event.scenePos().y() - self.rect.rect().y()
            )

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        dialog = QDialog(self.parent())
        dialog_layout = QVBoxLayout()
        top_line_edit = QLineEdit(parent=dialog)
        dialog_layout.addWidget(top_line_edit)

        def on_enter():
            self.text = top_line_edit.text()
            dialog.done(1)

        top_line_edit.editingFinished.connect(on_enter)
        dialog.setLayout(dialog_layout)
        dialog.exec_()
