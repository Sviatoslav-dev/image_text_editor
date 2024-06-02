from PySide2.QtWidgets import QMessageBox


class BaseController:
    def __init__(self, window, path, start_window):
        self.start_window = start_window
        self.selected_action = None
        self.window = window
        self.scene = window.scene
        self.model = None

    def show_not_found_message(self, msg_text="Text not found"):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Message")
        msg.setText(msg_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
