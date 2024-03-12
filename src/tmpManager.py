from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class tmpManager(QWidget):
    SIGNAL_start_training = pyqtSignal(str, str, str)

    def __init__(self):
        super(tmpManager, self).__init__()
        main_layout = QVBoxLayout()
        add_files_button = QPushButton("Load Data")
        add_files_button.setFixedSize(200, 25)
        add_files_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        main_layout.addWidget(add_files_button)
        main_layout.setAlignment(add_files_button, Qt.AlignCenter)
        main_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        _label = QLabel("Select from existing datasets:")
        main_layout.addWidget(_label)
        main_layout.setAlignment(_label, Qt.AlignCenter)

        self.setLayout(main_layout)