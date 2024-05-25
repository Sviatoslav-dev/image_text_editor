import cv2
import qimage2ndarray
from PySide2 import QtGui, QtCore
from PySide2.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QFileDialog

from base_controller import BaseController
from translation_dialog import TranslationDialog
from video_actions import VideoAction
from video_model import VideoModel


class VideoController(BaseController):
    pause = False
    first_stop = False

    def __init__(self, window, path, start_window):
        super().__init__(window, path, start_window)
        self.selected_action = VideoAction.ReplaceText
        self.scene = window.scene

        self.model = VideoModel(path)
        self.setup_camera(self.model.fps)
        # self.window.quit_button.clicked.connect(self.close_win)
        self.window.slider.valueChanged.connect(self.seek_video)
        QtCore.QObject.connect(
            self.window.play_pause_button, QtCore.SIGNAL("clicked()"), self.play_pause)
        self.scene.update_video = self.update_video

        self.window.replace_action.triggered.connect(self.set_action_replace)
        self.window.translate_text.triggered.connect(self.set_action_translate)
        self.window.remove_action.triggered.connect(self.set_action_remove)
        self.window.copy_action.triggered.connect(self.set_action_copy)
        self.window.find_action.triggered.connect(self.find_frame_with_text)
        self.window.save_action.triggered.connect(self.save_video)
        self.window.close_file_action.triggered.connect(self.close_file)
        self.window.select_whole.triggered.connect(self.set_action_translate)
        self.window.undo.triggered.connect(self.undo)

    def update_video(self):
        {
            VideoAction.ReplaceText: self.replace_text,
            VideoAction.RemoveText: self.remove_text,
            VideoAction.CopyText: self.copy_text,
            VideoAction.TranslateText: self.translate,

        }[self.selected_action]()

    def set_action_replace(self):
        self.selected_action = VideoAction.ReplaceText

    def set_action_remove(self):
        self.selected_action = VideoAction.RemoveText

    def set_action_copy(self):
        self.selected_action = VideoAction.CopyText

    def set_action_translate(self):
        dialog = TranslationDialog()
        dialog.exec_()
        self.model.translate_option = dialog.selected_option
        self.selected_action = VideoAction.TranslateText

    def setup_camera(self, fps):
        self.window.slider.setMinimum(0)
        self.window.slider.setMaximum(self.model.get_frames_count())

        self.window.frame_timer.timeout.connect(lambda: self.display_video_stream())
        self.window.frame_timer.start(int(1000 // fps))

    def show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, (self.window.video_size.width(), self.window.video_size.height()),
                           interpolation=cv2.INTER_AREA)
        image = qimage2ndarray.array2qimage(frame)
        self.window.frame_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def display_video_stream(self):
        ret, frame = self.model.next_frame()
        current_frame = self.model.get_current_frame_num()
        self.window.slider.setValue(current_frame)

        if not ret:
            self.play_pause()
            return False

        self.show_frame(frame)
        if not self.first_stop:
            self.play_pause()
            self.first_stop = True

    def play_pause(self):
        if not self.pause:
            self.window.frame_timer.stop()
            self.window.play_pause_button.setText("Play")
            # self.tracker = None
        else:
            self.window.frame_timer.start(int(1000 // self.model.fps))
            self.window.play_pause_button.setText("Pause")
            # self.video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.1)

        self.pause = not self.pause
        self.scene.pause = self.pause

    def change_size(self, from_, to, x):
        return x / from_ * to

    def seek_video(self):
        if self.model.video_capture:
            frame_number = self.window.slider.value()
            self.model.set_frame_number(frame_number)
            if self.pause and self.first_stop:
                ret, frame = self.model.next_frame()
                self.show_frame(frame)

    def _resize_rect(self):
        current_frame = self.model.get_current_frame()
        height, width = current_frame.shape[:2]

        x = int(self.change_size(self.window.video_size.width(), width, self.scene.rect.rect().x()))
        y = int(self.change_size(self.window.video_size.height(), height, self.scene.rect.rect().y()))
        rect_weight = int(self.change_size(self.window.video_size.width(), width, self.scene.rect.rect().width()))
        rect_height = int(self.change_size(self.window.video_size.height(), height, self.scene.rect.rect().height()))
        return x, y, rect_weight, rect_height

    def replace_text(self):
        dialog = QDialog(self.scene.parent())
        dialog_layout = QVBoxLayout()
        top_line_edit = QLineEdit(parent=dialog)
        dialog_layout.addWidget(top_line_edit)
        text = ""

        def on_enter():
            nonlocal text
            text = top_line_edit.text()
            dialog.done(1)

        top_line_edit.editingFinished.connect(on_enter)
        dialog.setLayout(dialog_layout)
        dialog.exec_()

        success = self.model.replace_text(text, *self._resize_rect())
        if not success:
            self.show_not_found_message()
        # self.video_model.replace_text_revert(text, *self._resize_rect())

        self.show_frame(self.model.get_current_frame())

    def remove_text(self):
        success = self.model.remove_text(*self._resize_rect())
        if not success:
            self.show_not_found_message()
        success = self.model.remove_text_revert(*self._resize_rect())
        if not success:
            self.show_not_found_message()
        self.show_frame(self.model.get_current_frame())

    def copy_text(self):
        self.model.copy_text(*self._resize_rect())

    def save_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.window, "Save Video As", "",
                                                   "AVI Files (*.avi);;All Files (*)",
                                                   options=options)

        if file_path:
            self.model.save_video(file_path)

    def find_frame_with_text(self):
        dialog = QDialog(self.scene.parent())
        dialog_layout = QVBoxLayout()
        top_line_edit = QLineEdit(parent=dialog)
        dialog_layout.addWidget(top_line_edit)
        text = ""

        def on_enter():
            nonlocal text
            text = top_line_edit.text()
            dialog.done(1)

        top_line_edit.editingFinished.connect(on_enter)
        dialog.setLayout(dialog_layout)
        dialog.exec_()

        frame_num = self.model.find_frame_with_text(text, 50)
        if frame_num is not None:
            self.model.set_frame_number(frame_num)
            ret, frame = self.model.next_frame()
            self.window.slider.setValue(frame_num)
            self.show_frame(frame)

    def close_file(self):
        self.model.video_capture.release()
        cv2.destroyAllWindows()
        self.window.close()
        self.start_window.show()

    def translate(self):
        languges = self.model.translate_option
        text = self.model.translate_text(*self._resize_rect(), *languges)
        if text is None:
            self.show_not_found_message()
        success = self.model.replace_text(text, *self._resize_rect())
        if not success:
            self.show_not_found_message()

        self.show_frame(self.model.get_current_frame())

    def undo(self):
        self.model.read_all_frames()
        self.show_frame(self.model.get_current_frame())

