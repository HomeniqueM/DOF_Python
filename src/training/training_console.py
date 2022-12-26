from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QDialog
from PyQt6 import QtGui
from threading import Thread

WINDOW_SIZE = (750,400)
WINDOW_TITLE_LABEL = "Console"

class TrainingConsole(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.console_text = ""
        self.line_number = 0

        self.__set_window_properties()
        self.__load_ui()
        self.__connect_slots()

        self.add_text_to_console("Começando treinamento em breve...")

    def __load_ui(self):
        self.__create_main_layouts()
        self.__create_status_label()
        self.__create_console_text_box()
        self.__create_close_training_button()

    def __set_window_properties(self):
        self.setWindowTitle(WINDOW_TITLE_LABEL)
        self.setObjectName("MainWindow")
        width, height = WINDOW_SIZE
        self.resize(width,height)
        self.setFixedSize(width,height)

    def __create_main_layouts(self):
        self.main_vertical_layout = QtWidgets.QVBoxLayout(self)
        self.main_vertical_layout.setObjectName("main_vertica_layout")

    def __create_status_label(self):
        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.update_training_status("Leitura das Imagens")
        
        self.main_vertical_layout.addWidget(self.status_label)
        
    def __create_console_text_box(self):
        self.console = QtWidgets.QTextEdit()
        self.console.setObjectName("console_box")
        self.console.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,QtWidgets.QSizePolicy.Policy.Expanding)
        self.console.setReadOnly(True)
        self.console.setStyleSheet("color: white;"
                                    "background-color: black;"
                                    "selection-color: yellow;"
                                    "selection-background-color: green;");

        self.main_vertical_layout.addWidget(self.console)

    def __create_close_training_button(self):
        self.close_training_button = QtWidgets.QPushButton()
        self.close_training_button.setText("Fechar Treinamento")
        self.close_training_button.setObjectName("close_training_button")
        self.close_training_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,QtWidgets.QSizePolicy.Policy.Minimum)
        
        self.main_vertical_layout.addWidget(self.close_training_button)

    def __connect_slots(self):
        QtCore.QMetaObject.connectSlotsByName(self)

        self.close_training_button.clicked.connect(self.on_close_training_button)
    def on_close_training_button(self):
        self.close()

    def add_text_to_console(self, text):
        self.console_text += f'{text}\n'
        self.console.setText(self.console_text)
        self.line_number += 1
        cursor = QtGui.QTextCursor(self.console.document().findBlockByLineNumber(self.line_number))
        self.console.setTextCursor(cursor)

    def update_training_status(self, status):
        self.status_label.setText(f"Status do treinamento: {status}")

    def start(self, training_thread: Thread):
        self.exec()

        try:
            training_thread._stop()
        except Exception:
            pass 
        




