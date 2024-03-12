from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class TestingManager(QWidget):
    SIGNAL_start_testing = pyqtSignal(str)

    def __init__(self):
        super(TestingManager, self).__init__()
        main_layout = QVBoxLayout()
        test_button = QPushButton("Test")
        test_button.clicked.connect(self.start_testing)
        main_layout.addWidget(QLabel("Select test directory"))
        main_layout.addWidget(test_button)
        self.setLayout(main_layout)

        self.raise_()
        self.show()

    
    def start_testing(self):
        self.test_dir = QFileDialog.getExistingDirectory(self, "Select Test Directory")
        self.SIGNAL_start_testing.emit(self.test_dir)
