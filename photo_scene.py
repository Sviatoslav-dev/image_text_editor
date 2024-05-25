from PySide2.QtGui import *
from PySide2.QtWidgets import *


class PhotoScene(QGraphicsScene):
    def __init__(self, parent=None, pixmap_item=None):
        super(PhotoScene, self).__init__(parent)
        self.q_pixmap: QPixmap = pixmap_item
        self.update_image = None

    def mousePressEvent(self, event):
        print(event.scenePos().x(), event.scenePos().y())
        self.rect.setRect(event.scenePos().x(), event.scenePos().y(), 0, 0)
        super(PhotoScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.rect.setRect(
            self.rect.rect().x(),
            self.rect.rect().y(),
            event.scenePos().x() - self.rect.rect().x(),
            event.scenePos().y() - self.rect.rect().y()
        )

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.update_image()
        self.rect.setRect(
            0,
            0,
            0,
            0
        )
