from PySide2.QtCore import Qt
from PySide2.QtWidgets import QVBoxLayout, QDialog, QPushButton, QComboBox, QHBoxLayout, QLabel


class TranslationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_option = None
        self.setWindowTitle("Select languages")

    @property
    def languages_map(self):
        return {
            "English": "en",
            "Ukrainian": "uk",
        }

    def initUI(self):
        main_layout = QVBoxLayout()

        title_label = QLabel("Select Languages", self)
        title_label.setAlignment(Qt.AlignCenter)

        combo_layout = QHBoxLayout()

        self.combo_from = QComboBox(self)
        self.combo_from.addItems(["English", "Ukrainian"])

        self.label_to = QLabel("to", self)

        self.combo_to = QComboBox(self)
        self.combo_to.addItems(["Ukrainian", "English"])

        self.btn_select = QPushButton("Select", self)
        self.btn_select.clicked.connect(self.select_languages)

        combo_layout.addWidget(self.combo_from)
        combo_layout.addWidget(self.label_to)
        combo_layout.addWidget(self.combo_to)

        main_layout.addWidget(title_label)
        main_layout.addLayout(combo_layout)
        main_layout.addWidget(self.btn_select)
        self.setLayout(main_layout)

    def select_languages(self):
        from_lang = self.combo_from.currentText()
        to_lang = self.combo_to.currentText()
        self.selected_option = (self.languages_map[from_lang], self.languages_map[to_lang])
        self.accept()
