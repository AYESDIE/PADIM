import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ImageBrowserManager(QWidget):
    def __init__(self, image_folder):
        super().__init__()

        self.current_index = 0
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if
                            file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        self.init_ui()
        self.raise_()
        self.show()

    def init_ui(self):
        self.setWindowTitle('Image Browser')

        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)

        self.btn_previous = QPushButton('Previous', self)
        self.btn_previous.clicked.connect(self.show_previous_image)

        self.btn_next = QPushButton('Next', self)
        self.btn_next.clicked.connect(self.show_next_image)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label_image)
        layout.addWidget(self.btn_previous)
        layout.addWidget(self.btn_next)

        self.setLayout(layout)

        self.show_current_image()

    def show_previous_image(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.show_current_image()

    def show_next_image(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.show_current_image()

    def show_current_image(self):
        if self.image_files:
            image_path = self.image_files[self.current_index]
            pixmap = self.get_scaled_pixmap(image_path)
            self.label_image.setPixmap(pixmap)

    def get_scaled_pixmap(self, image_path):
        max_width = self.label_image.width()
        max_height = self.label_image.height()

        image = QImage(image_path)
        scaled_image = image.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        return QPixmap.fromImage(scaled_image)
