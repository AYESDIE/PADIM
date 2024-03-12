from src.ModelManager import ModelManager
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelManager()
    sys.exit(app.exec_())
