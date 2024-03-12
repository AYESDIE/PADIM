from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class TrainingManager(QWidget):
    SIGNAL_start_training = pyqtSignal(str, str, str)

    def __init__(self):
        super(TrainingManager, self).__init__()
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
        add_files_button.clicked.connect(self.load_data)

        horizontal_layout = QHBoxLayout()
        horizontal_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.setAlignment(horizontal_layout, Qt.AlignCenter)
        main_layout.addLayout(horizontal_layout)
        _ = QVBoxLayout()
        bottle_button = QPushButton()
        bottle_button.setIcon(QIcon("assets/images/bottle.png"))
        bottle_button.setIconSize(QSize(100, 100))
        bottle_button.setFixedSize(100, 100)
        bottle_button.clicked.connect(self.load_bottle)
        _label = QLabel("Bottle")
        _.addWidget(bottle_button)
        _.addWidget(_label)
        _.setAlignment(bottle_button, Qt.AlignCenter)
        _.setAlignment(_label, Qt.AlignCenter)
        horizontal_layout.addLayout(_)

        _ = QVBoxLayout()
        cable_button = QPushButton()
        cable_button.setIcon(QIcon("assets/images/cable.png"))
        cable_button.setIconSize(QSize(100, 100))
        cable_button.setFixedSize(100, 100)
        cable_button.clicked.connect(self.load_cable)
        _label = QLabel("Cable")
        _.addWidget(cable_button)
        _.addWidget(_label)
        _.setAlignment(cable_button, Qt.AlignCenter)
        _.setAlignment(_label, Qt.AlignCenter)
        horizontal_layout.addLayout(_)
        horizontal_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.setLayout(main_layout)


    def load_data(self):
        self.data_dir = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        self.model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        self.start_training()
    
    def load_bottle(self):
        self.data_dir = "./data/bottle/train"
        self.model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        self.start_training()

    def load_cable(self):
        self.data_dir = "./data/cable/train/good"
        self.model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        self.start_training()
    
    def start_training(self):
        self.SIGNAL_start_training.emit(self.data_dir, self.model_dir, "UUID-1")