from PySide2.QtWidgets import QVBoxLayout, QDialog, QPushButton


class TranslationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_option = None

    def initUI(self):
        layout = QVBoxLayout()

        btn_eng_to_ukr = QPushButton("English to Ukrainian", self)
        btn_eng_to_ukr.clicked.connect(self.select_eng_to_ukr)

        btn_ukr_to_eng = QPushButton("Ukrainian to English", self)
        btn_ukr_to_eng.clicked.connect(self.select_ukr_to_eng)

        layout.addWidget(btn_eng_to_ukr)
        layout.addWidget(btn_ukr_to_eng)
        self.setLayout(layout)

        self.setWindowTitle("Select Translation Direction")

    def select_eng_to_ukr(self):
        self.selected_option = ("en", "uk")
        self.accept()

    def select_ukr_to_eng(self):
        self.selected_option = ("uk", "en")
        self.accept()
