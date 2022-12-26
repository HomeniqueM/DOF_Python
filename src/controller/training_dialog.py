from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QDialog, QComboBox
from PyQt6.QtGui import QPixmap, QImage

from utils import utils_os as uo
from utils import qt_utils as QtUtils
from model.model_type import ModelType
from model.training_type import TrainingType
from model.model_info import parse_model_info, ModelInfo
from model.training_info import TrainingInfo
from utils import image_utils as ImageUtils

import numpy as np

CACHE_FOLDER = ".cache"

WINDOW_SIZE = (750,630)

"""
Textos da tela
"""

# Titulo da janela
WINDOW_TITLE_LABEL = "Preparar treinamento"

# Modelo da IA
IA_MODEL_LABEL = "Escolha técnica de classificação"

# Escolher diretorio de treinamento
CHOOSE_DIRECTORY_LABEL = "Diretório das imagens para treino"
SEARCH_DIRECTORY_LABEL = "Procurar"

# Operacoes nas imagens
EQUALIZE_IMAGES_LABEL = "Equalizar imagens"
MIRROR_IMAGES_LABEL = "Espelhar horizontalmente"

# Opcoes finais
TRAIN_LABEL = "Fazer Treinamento"
CANCEL_LABEL = "Cancelar"

# Pre-visualizacao da imagem
PREVIEW_LABEL = "Preview"

# Campo da cache
CACHE_LABEL = "Escolha um arquivo de cache"
DEFAULT_CACHE_OPTION = "Selecione a cache"

class TrainingDialog(QDialog):

    def __init__(self, parent=None,model=None):
        super().__init__(parent)
        if model == None:
            model = ModelType.SVM

        self.model = model
        self.example_image = ImageUtils.open_example_image()
        self.curr_image = None
        self.training_type = TrainingType.Binary

        self.__load_ui()
        
        self.equalization = False
        self.horizontal_mirror = False
        self.operation_canceled = True
        self.training_path_text = None

    def start_dialog(self):

        self.exec()

        data = TrainingInfo(ModelInfo(self.model,
                            self.equalization,
                            self.horizontal_mirror),
                            self.training_type,
                            self.training_path_text)

        return (self.operation_canceled, data)

    def __load_ui(self):
        self.setWindowTitle(WINDOW_TITLE_LABEL)
        self.setObjectName("MainWindow")
        width, height = WINDOW_SIZE
        self.resize(width,height)
        self.setFixedSize(width,height)

        self.__create_main_layout()

        self.__create_cache_combo_box()

        self.__create_training_type_combo_box()

        self.__create_model_combo_box()

        self.__add_spacing()

        self.__create_training_layout()

        self.__add_spacing()

        self.__create_equalization_checkbox()

        self.__create_horizontal_mirror_checkbox()

        self.__create_image_layout()

        self.__add_expanding_spacing()

        self.__create_training_and_cancel_button()

        self.__connect_slots()


    def __create_main_layout(self):
        self.main_vertical_layout = QtWidgets.QVBoxLayout(self)
        self.main_vertical_layout.setObjectName("main_vertica_layout")

    def __create_training_type_combo_box(self):
        self.training_type_label = QtWidgets.QLabel()
        self.training_type_label.setObjectName("training_type_label")
        self.training_type_label.setText("Escolha o tipo de treinamento de classificação da imagem")
        self.main_vertical_layout.addWidget(self.training_type_label)

        self.training_type_combo_box = QComboBox()
    
        self.training_type_combo_box.addItems(TrainingType.list())
        self.training_type_combo_box.setCurrentText(self.training_type.value)
        self.main_vertical_layout.addWidget(self.training_type_combo_box)

    def __create_cache_combo_box(self):
        self.cache_label = QtWidgets.QLabel()
        self.cache_label.setObjectName("cache_label")
        self.cache_label.setText(CACHE_LABEL)
        self.main_vertical_layout.addWidget(self.cache_label)

        self.cache_combo_box = QComboBox()

        cache_files = [DEFAULT_CACHE_OPTION]
        cache_files.extend(uo.files_in_path(CACHE_FOLDER))
    
        self.cache_combo_box.addItems(cache_files)
        self.main_vertical_layout.addWidget(self.cache_combo_box)

    def __create_model_combo_box(self):
        self.model_type_label = QtWidgets.QLabel()
        self.model_type_label.setObjectName("model_type_label")
        self.model_type_label.setText(IA_MODEL_LABEL)
        self.main_vertical_layout.addWidget(self.model_type_label)

        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(ModelType.list())
        self.model_combo_box.setCurrentText(self.model.value)
        self.model_combo_box.setStyleSheet(QtUtils.disabled_stylesheet(self.model_combo_box))
        self.main_vertical_layout.addWidget(self.model_combo_box)

    def __add_spacing(self):
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.main_vertical_layout.addItem(spacerItem)

    def __add_expanding_spacing(self):
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.main_vertical_layout.addItem(spacerItem)

    def __create_training_layout(self):
        self.training_layout = QtWidgets.QHBoxLayout()
        self.training_layout.setObjectName("training_layout")

        # Training path label
        self.training_path_label = QtWidgets.QLabel()
        self.training_path_label.setObjectName("training_path_label")
        self.training_path_label.setText(CHOOSE_DIRECTORY_LABEL)
        self.training_layout.addWidget(self.training_path_label)

        # Path inserting
        self.training_images_path = QtWidgets.QLineEdit()
        self.training_images_path.setEnabled(False)
        self.training_images_path.setObjectName("training_images_path")

        # Search folder button
        self.search_folder_button = QtWidgets.QPushButton()
        self.search_folder_button.setObjectName("search_folder_button")
        self.search_folder_button.setText(SEARCH_DIRECTORY_LABEL)
        self.search_folder_button.setStyleSheet(QtUtils.disabled_stylesheet(self.search_folder_button))

        # Insert Training widgets into layouts
        self.training_layout.addWidget(self.training_images_path)
        self.training_layout.addWidget(self.search_folder_button)

        # Add training layout to main layout
        self.main_vertical_layout.addLayout(self.training_layout)

    def __create_equalization_checkbox(self):
        self.equalization_checkbox = QtWidgets.QCheckBox(text=EQUALIZE_IMAGES_LABEL)
        self.equalization_checkbox.setObjectName("equalization_checkbox")
        self.equalization_checkbox.setChecked(False)
        self.equalization_checkbox.setStyleSheet(QtUtils.disabled_stylesheet(self.equalization_checkbox,True))
        self.main_vertical_layout.addWidget(self.equalization_checkbox)     

    def __create_horizontal_mirror_checkbox(self):
        self.horizontal_mirror_checkbox = QtWidgets.QCheckBox(text=MIRROR_IMAGES_LABEL)
        self.horizontal_mirror_checkbox.setObjectName("horizontal_mirror_checkbox")
        self.horizontal_mirror_checkbox.setChecked(False)
        self.horizontal_mirror_checkbox.setStyleSheet(QtUtils.disabled_stylesheet(self.horizontal_mirror_checkbox,True))
        self.main_vertical_layout.addWidget(self.horizontal_mirror_checkbox)  

    def __create_image_layout(self):
        self.image_layout = QtWidgets.QVBoxLayout()
        self.image_layout.setObjectName("image_layout")

        image = QImage(self.example_image, 
                        self.example_image.shape[1],
                        self.example_image.shape[0],
                        self.example_image.strides[0],
                        QImage.Format.Format_RGB888)
        
        image_container = QPixmap.fromImage(image)
        self.image_box = QtWidgets.QLabel()
        self.image_box.setObjectName("image_box")
        self.image_box.setPixmap(image_container)
        self.image_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setObjectName("image_label")
        self.image_label.setText(PREVIEW_LABEL)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.image_layout.addWidget(self.image_box)
        self.image_layout.addWidget(self.image_label)

        self.main_vertical_layout.addLayout(self.image_layout)

    def __create_training_and_cancel_button(self):
        self.training_button = QtWidgets.QPushButton(self)
        self.training_button.setObjectName("training_button")
        self.training_button.setText(TRAIN_LABEL)

        self.cancel_button = QtWidgets.QPushButton(self)
        self.cancel_button.setObjectName("cancel_button")
        self.cancel_button.setText(CANCEL_LABEL)

        self.training_cancel_layout = QtWidgets.QHBoxLayout()
        self.training_cancel_layout.setObjectName("training_cancel_layout")
        self.training_cancel_layout.addWidget(self.training_button)
        self.training_cancel_layout.addWidget(self.cancel_button)

        self.main_vertical_layout.addLayout(self.training_cancel_layout)

    def __connect_slots(self):
        QtCore.QMetaObject.connectSlotsByName(self)

        self.cancel_button.clicked.connect(self.cancel_operation)
        self.training_button.clicked.connect(self.start_training)
        self.search_folder_button.clicked.connect(self.search_training_folder)
        self.model_combo_box.currentTextChanged.connect(self.on_model_combo_box_text_changed)
        self.cache_combo_box.currentTextChanged.connect(self.on_cache_combo_box_text_changed)
        self.equalization_checkbox.toggled.connect(self.set_equalization)
        self.horizontal_mirror_checkbox.toggled.connect(self.set_horizontal_mirror)
        self.training_type_combo_box.currentTextChanged.connect(self.on_training_type_text_changed)

    def on_training_type_text_changed(self, text):
        self.training_type = TrainingType(text)

    def cancel_operation(self):
        self.close()

    def on_model_combo_box_text_changed(self,text):
        self.model = ModelType(text)
        self.model_combo_box.setCurrentIndex(ModelType.enum_to_int(self.model))

    def on_cache_combo_box_text_changed(self,text):
        
        if text == DEFAULT_CACHE_OPTION:

            self.__toggle_interface_enabled(True)
            self.horizontal_mirror_checkbox.setChecked(False)
            self.equalization_checkbox.setChecked(False)
            self.training_images_path.setText("")

        else:

            self.__toggle_interface_enabled(False)

            model_info = parse_model_info(text)

            self.model_combo_box.setCurrentIndex(ModelType.enum_to_int(model_info.type))
            self.equalization_checkbox.setChecked(model_info.equalize_images)
            self.horizontal_mirror_checkbox.setChecked(model_info.horizontal_mirror)
            self.training_images_path.setText(text)

    def __toggle_interface_enabled(self, value):
        self.model_combo_box.setEnabled(value)
        self.equalization_checkbox.setEnabled(value)
        self.horizontal_mirror_checkbox.setEnabled(value)
        self.search_folder_button.setEnabled(value)

    def set_equalization(self,equalize):
        self.equalization = equalize
        self.__update_image_pixmap()


    def set_horizontal_mirror(self,horizontal_mirror):
        self.horizontal_mirror = horizontal_mirror
        self.__update_image_pixmap()


    def start_training(self):

        if not self.training_images_path.text():
            msg = QtWidgets.QMessageBox()

            msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            msg.setText("Erro ao iniciar o treinamento!")
            msg.setInformativeText("Caminho para imagens de treinamento não foi informado")
            msg.setWindowTitle("Erro!")

            msg.exec()
            return

        self.training_path_text = self.training_images_path.text()
        self.operation_canceled = False

        self.close()

    def __update_image_pixmap(self):

        self.curr_image = self.example_image.copy()

        if self.equalization:
            self.curr_image = ImageUtils.equalization(self.curr_image)
        if self.horizontal_mirror:
            self.curr_image = ImageUtils.mirror_horizontally(self.curr_image)

        self.curr_image = QImage(self.curr_image, 
                        self.example_image.shape[1],
                        self.example_image.shape[0],
                        self.example_image.strides[0],
                        QImage.Format.Format_RGB888)
        
        
        image_container = QPixmap.fromImage(self.curr_image)
        self.image_box.setPixmap(image_container)

    def set_training_images_folder_path(self,file_path):
        self.training_images_path.setText(file_path)

    def search_training_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 
                                                "Abrir imagens de treino",
                                                uo.get_src_projec())

        if not path:
            return

        folders_in_path = uo.folders_in(path)

        if not self.folder_naming_are_a_match(folders_in_path) or not folders_in_path:
            msg = QtWidgets.QMessageBox()

            msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            msg.setText("Erro ao carregar as imagens de treinamento")
            msg.setInformativeText("Por favor, escolha uma pasta contendo: \'auto_test\',\'test\',\'train\',\'val\'.")
            msg.setWindowTitle("Erro!")

            msg.exec()
            return

        self.training_images_path.setText(path)
    # Criteiro de validação das pastas 
    def folder_naming_are_a_match(self,folders_in_path):
        folders_names = ['auto_test','test','train','val']

        return np.array_equal(folders_in_path,folders_names)