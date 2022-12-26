import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.io import imread

from xgboost import XGBClassifier


import os
from PIL import Image, ImageOps
import utils.cache_utils as uc
from utils.stopwatch import Stopwatch
import utils.utils_os as ou
import time

from model.model_type import ModelType
from PyQt6.QtCore import pyqtSignal as Signal, QObject

class TrainingXGBoost (QObject):
    # Atribuição dos sinais do QT
    new_information = Signal(str)
    new_status = Signal(str)
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Criando o classificador XGBoost
        self._params = {'objective': ['reg:squarederror'],
                        'max_depth': [3,8],
                        'colsample_bylevel': [0.5],
                        'learning_rate': [0.1],
                        'random_state': [20],
                        'n_estimators': [100],
                        'colsample_bytree': [0.5,0.7],
                        'subsample': [0.6, 1]}
        self.__classifier = XGBClassifier()

        # Qual modelo vai ser utilizado para o treinamento
        self.__model = None

        # Classes de imagens
        self.__kl_classes = ['0', '1', '2', '3', '4']

        self.__confusion_matrix = None
        self.__accuracy = 0
        self.__specificity = 0
        self.__f1_score = 0
        self.__precision_score = 0
        self.__sensitivity = 0
    

        # Tempo de processamento
        self.__times = {
            "training": 0,
            "processing": 0,
        }

    def get_confusion_matrix(self):
        return self.__confusion_matrix

    def get_accuracy(self):
        return self.__accuracy

    def get_specificity(self):
        return self.__specificity

    def get_metrics(self):
        return {
            "accuracy" : self.__accuracy,
            "specificity" : self.__specificity,
            "sensibility" : self.__sensitivity,
            "precision": self.__precision_score,
            "f1" : self.__f1_score
        }

    def get_times(self):
        return self.__times

    def __extract_image_texture_descriptors(self, image):

        characteristics = []

        # Corrigir

        glcm = graycomatrix(np.array(image),
                            [1, 2, 4, 8, 16],
                            [0, np.pi / 2, np.pi / 4, np.pi / 8, 3 *
                                np.pi / 4, 5 * np.pi / 8, 7 * np.pi/8],
                            symmetric=False,
                            normed=True, levels=32)

        characteristics.append(shannon_entropy(glcm, base=2))
        characteristics.append(graycoprops(glcm, "energy"))
        characteristics.append(graycoprops(glcm, "correlation"))
        characteristics.append(graycoprops(glcm, "dissimilarity"))
        characteristics.append(graycoprops(glcm, "homogeneity"))

        characteristics = np.concatenate(characteristics, axis=None)

        return characteristics

        # Recupera todas as imagens dentro do diretorio informado

    # É recebido um diretorio raiz onde o a função vai recuperar e trata cada imagem
    # uma por vez de forma a gerar os descritores e a classes KL

    def read_image_and_process(self, path, mirrored=False, equalization=False):
        texture = []
        kl = []
        root_image_folder = ou.get_all_subfolder(path)

        for i, folder in enumerate(root_image_folder):
          #  images = self.__extract_images_from_folder( os.path.join(path, folder))
            images_paths = ou.get_images_from_path(os.path.join(path, folder))

            j = 0
            l = len(images_paths)

            for file in images_paths:
                image = Image.open(file)

                j += 1
                self.new_information.emit(
                    f'kl #{i}| Imagem #{j:04}/{l:04} ({round((j/l)*100,1):02}%)')

                if equalization:
                    equalization_img = ImageOps.equalize(image)

                    # Definido o numero de tons de cinza 32
                    equalization_img = equalization_img.quantize(32)

                    equali_desc = self.__extract_image_texture_descriptors(
                        equalization_img)
                    texture.append(equali_desc)
                    kl.append(self.__kl_classes.index(self.__kl_classes[i]))

                if mirrored:
                    mirror_img = ImageOps.mirror(image)
                    mirror_img = mirror_img.quantize(32)
                    mirror_desc = self.__extract_image_texture_descriptors(
                        mirror_img)
                    texture.append(mirror_desc)
                    kl.append(self.__kl_classes.index(
                        self.__kl_classes[i]))

                image = image.quantize(32)
                descriptor = self.__extract_image_texture_descriptors(image)
                texture.append(descriptor)
                kl.append(self.__kl_classes.index(self.__kl_classes[i]))

        self.new_information.emit(f'Imagens para treino {len(texture)}')
        return (texture, kl)

    # Função para chamar o treino e o teste de forma automatica

    def setup(self, root_path, parameters=None, mirrored=False, equalization=False, model=None):

        if str.__contains__(root_path,'.data'):
            root_path = os.path.join( os.getcwd(),'.cache',root_path)
            self.new_information.emit(f'Load model from:\n{root_path}')
            self.load_Model(root_path)
            data_frame_test = self.load_dataFrame(root_path)
            self.test_classifier_data_frame(data_frame_test)
        else:
            self.train_classifier(
                root_path, mirrored=mirrored, equalization=equalization)

            self.test_classifier(root_path, 'test')

    # Treina o modelo dado os paramentros
    def __feed_classifier(self, x_axis, y_axis):
        self.new_status.emit("Treinamento Inciado")
        self.__model.fit(x_axis, y_axis)

        self.new_status.emit("Treinamento Finalizado")

    # Aplica a tratativa para a extração de caracterista das images e as testas com o modelo treinado
    def test_classifier(self, root_path, name_folder):
        
        training_timer = Stopwatch()
        training_timer.start()

        self.new_status.emit("[Validado testes] ")
        texture_test = []
        kl_test = []
        test_path = os.path.join(root_path, name_folder)

        texture_test, kl_test = self.read_image_and_process(test_path)

        data_frame_test = pd.DataFrame(np.array(texture_test))
        data_frame_test['Target'] = np.array(kl_test)

        x_axis_test = data_frame_test.iloc[:, :-1]
        y_axis_test = data_frame_test.iloc[:, -1]

        y_pred = self.__model.predict(x_axis_test)

        self.__accuracy = "{:.2f}".format(
            accuracy_score(y_axis_test, y_pred) * 100)

        self.__specificity = (100 - float(self.__accuracy))/300

        self.__confusion_matrix = confusion_matrix(y_axis_test, y_pred)

        self.__accuracy = "{:.2f}".format(
            accuracy_score(y_axis_test, y_pred) * 100)

        self.__f1_score = "{:.2f}".format(f1_score(y_axis_test, y_pred, average='weighted'))

        self.__precision_score = "{:.2f}".format(precision_score(y_axis_test, y_pred,average='weighted'))

        self.__sensitivity = "{:.2f}".format(recall_score(y_axis_test, y_pred,average='weighted'))

        self.__confusion_matrix = confusion_matrix(y_axis_test, y_pred)
        self.__times['time_to_train'] = training_timer.get_time()

        uc.save_images_data_frame(data=data_frame_test,
                                  path=root_path, suffix=name_folder)

    def train_classifier(self, root_path,  mirrored=False, equalization=False):
        self.new_information.emit("XGBoost")
        start = time.time()

        # iniciado valores
        texture_train = []
        kl_train = []

        train_path = os.path.join(root_path, 'train')
        self.new_information.emit("Leitura de imagens e extração de texturas")
        texture_train, kl_train = self.read_image_and_process(
            train_path, mirrored, equalization)
        self.new_information.emit("Preparação para o treinamento")
        data_frame_train = pd.DataFrame(np.array(texture_train))
        data_frame_train['Target'] = np.array(kl_train)

        x_axis_train = data_frame_train.iloc[:, :-1]
        y_axis_train = data_frame_train.iloc[:, -1]

        self.__model = GridSearchCV(
            self.__classifier, self._params, n_jobs=10)

        self.__feed_classifier(x_axis_train, y_axis_train)

        self.new_information.emit(f'Parametros Selecionados: {self.__model.best_params_}')
        end = time.time()
        self.__times['training'] = end-start
        self.new_information.emit(f'Tempo de treinamento: {end-start}')
        uc.save(model_name=ModelType.XGBOOST.value, data=self.__model, type_ouput='data',
                path=root_path, suffix='train', equalization=equalization, mirrored=mirrored)
        uc.save(model_name=ModelType.XGBOOST.value, data=self.__times, type_ouput='txt',
                path=root_path, suffix='train', equalization=equalization, mirrored=mirrored)
    
    def predict_kl_image(self, image):
        training_timer = Stopwatch()
        training_timer.start()

        texture_test = []
        kl_test = []

        new_image       = Image.open(image)
        new_image       = new_image.quantize(32)

        texture_test     = self.__extract_image_texture_descriptors(new_image)

        data_frame_test = pd.DataFrame(np.array(texture_test))

        x_axis_test = data_frame_test.iloc[:, :-1]

        y_pred = self.__model.predict(x_axis_test)

        print(y_pred)
        self.__times['prediction'] = training_timer.get_time()

    # Dada uma imagem é testado com o modelo treinado
    def predict_kl_image1(self, image):
        start = time.time()

        new_image       = Image.open(image)
        new_image       = new_image.quantize(32)

        descriptors     = self.__extract_image_texture_descriptors(new_image)

        kl_class = self.__model.predict(descriptors)

        end = time.time()

        self.__times['prediction'] = end-start

        return self.__kl_classes[kl_class]

    # carrega o modelos e as informações de tempo
    def load_Model(self, cache_path):
        self.__times = pd.read_pickle (cache_path.replace('.data','.txt'))
        self.__model = pd.read_pickle(cache_path)
    
    def load_dataFrame( self, cache_path):
        print(f'cache_path: {cache_path}')
        name_to_replace = cache_path.split('/')[-1]
        new_name='' 
        if str.__contains__(name_to_replace,'299'):
            new_name= 'kneeKL299_test.data'
        else:
            new_name= 'kneeKL224_test.data'

        return pd.read_pickle (cache_path.replace(name_to_replace,new_name))

    def test_classifier_data_frame(self, data_frame_test ):
        training_timer = Stopwatch()
        training_timer.start()

        x_axis_test = data_frame_test.iloc[:, :-1]
        y_axis_test = data_frame_test.iloc[:, -1]

        y_pred = self.__model.predict(x_axis_test)


        self.__accuracy = "{:.2f}".format(
            accuracy_score(y_axis_test, y_pred) * 100)

        self.__specificity = (100 - float(self.__accuracy))/  len(x_axis_test)

        self.__f1_score = "{:.2f}".format(f1_score(y_axis_test, y_pred, average='weighted'))

        self.__precision_score = "{:.2f}".format(precision_score(y_axis_test, y_pred,average='weighted'))

        self.__sensitivity = "{:.2f}".format(recall_score(y_axis_test, y_pred,average='weighted'))

        self.__confusion_matrix = confusion_matrix(y_axis_test, y_pred)

        self.new_information.emit(f'Concluido o teste')
 
        self.new_information.emit(f'Acurácia: {self.__accuracy}')
        self.new_information.emit(f'Especificidade: {self.__specificity}')
        self.new_information.emit(f'Matriz de confusão: {self.__confusion_matrix}')
        self.new_information.emit(f'Tempo de treinamento: {training_timer.get_time()}s')
        self.new_information.emit(f'Melhores Parametros = {self.__model.best_params_}')

        self.__times['time_to_train'] = training_timer.get_time()

        self.new_status.emit("Treinamento completo")