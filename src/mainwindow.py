# Trabalho Prático - Processamento e Análise de Imagens
# Bryan Santos e Homenique Martins
# 02/2022
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import pyqtSignal as Signal

from controller.training_dialog import TrainingDialog
from training.training_cnn import GoogleNet
from training.training_console import TrainingConsole
from training.training_svm import trainingSVM
#from training.training_thunder_svm import trainingThuderSVM
from training.training_xgboost import TrainingXGBoost
from window import Ui_MainWindow
from model.model_type import ModelType
from functools import partial
from threading import Thread


class MainWindow(QMainWindow, Ui_MainWindow):

    training_completed = Signal(list,'QVariantHash')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__classifier = None
        self.setupUi(self)

        #self.training_completed.connect(self.widget_confusion_matrix.on_training_completed)

        self.setWindowTitle("Trabalho Prático - DOF")

        # Conexões de botões
        self.actionAbrir.triggered.connect(
            self.image_view.imageControle.open_image)
        self.actionSalvar.triggered.connect(
            self.image_view.imageControle.save_image)
        self.actionBusca_de_sub_regiao.triggered.connect(
            self.image_view.imageControle.find_sub_image)
        self.actionRecorte.triggered.connect(
            self.image_view.imageControle.crop_image)

        # Conexoes para abrir o treinamento
        self.actionGOOGLENET.triggered.connect(
            partial(self.open_controller_training, ModelType.GOOGLENET))
        self.actionSVM.triggered.connect(
            partial(self.open_controller_training, ModelType.SVM))
        self.actionXGBoost1.triggered.connect(
            partial(self.open_controller_training, ModelType.XGBOOST))

    def fazer_previsao(self):

        image = self.image_view.imageControle.image_path

        kl_class = self.__classifier.predict_kl_image(image)

        times = self.__classifier.get_times()

        prediction_time = times["prediction"]

        msg = QMessageBox()

        msg.setText("Uma previsão foi feita!")
        msg.setInformativeText(f"A classe KL prevista foi: {kl_class}")
        msg.setDetailedText(f"Tempo previsto: {prediction_time}s")
        msg.setWindowTitle("Previsão KL")

        msg.exec()



    def open_controller_training(self, model):

        training_console = TrainingConsole()
        
        ui = TrainingDialog(self, model=model)

        # Chamar tela de configuracao de treinamento
        operation_canceled, training_data = ui.start_dialog()

        # Se o usuario cancelar o treinamento
        if operation_canceled:
            return

        # TODO: Logica pra fazer treinamento da IA
        print(training_data)

        training = Thread(target=self.start_training,args=(training_data,training_console))

        training.start()

        training_console.start(training)

    def start_training(self,training_data, training_console):
        
        if training_data.model.type == ModelType.SVM:
            self.__classifier        = trainingSVM()
            self.__classifier.new_information.connect(training_console.add_text_to_console)
            self.__classifier.new_status.connect(training_console.update_training_status)
            self.__classifier.setup(training_data.training_images_path,mirrored=training_data.model.horizontal_mirror, equalization=training_data.model.equalize_images)

        if training_data.model.type == ModelType.GOOGLENET :
            self.__classifier   = GoogleNet()
            self.__classifier.new_information.connect(training_console.add_text_to_console)
            self.__classifier.new_status.connect(training_console.update_training_status)
            self.__classifier.setup(training_data.training_images_path,mirrored=training_data.model.horizontal_mirror, equalization=training_data.model.equalize_images)

        if training_data.model.type == ModelType.XGBOOST :
            self.__classifier   = TrainingXGBoost()
            self.__classifier.new_information.connect(training_console.add_text_to_console)
            self.__classifier.new_status.connect(training_console.update_training_status)
            self.__classifier.setup(training_data.training_images_path,mirrored=training_data.model.horizontal_mirror, equalization=training_data.model.equalize_images)
                                
        self.training_completed.emit(self.__classifier.get_confusion_matrix().tolist(),
                            self.__classifier.get_metrics())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())