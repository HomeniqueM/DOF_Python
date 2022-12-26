import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.io import imread
from PIL import Image,ImageOps,ImageEnhance
import cv2 as cv
from model.model_type import ModelType
import utils.utils_os as os_utils
from utils.stopwatch import Stopwatch
import os
import utils.cache_utils as uc

from PyQt6.QtCore import pyqtSignal as Signal, QObject


class trainingSVM(QObject):

    new_information = Signal(str)
    new_status = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Ativar para preverresultados
        self.__classifier = svm.SVC(probability=True)
        # Qual modelo vai ser utilizado para o treinamento
        self.__model = None
        # Hyper-paramenter para sem testados
        # C:
        # gamma :
        # Kernel: função de ajuste
        self.__param_grid = {'C': [1.4], 'gamma': ['auto'], 'kernel': ['linear']}
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
                            [8, 16, 32],
                            [0,
                            np.pi / 8,
                            np.pi / 2, 
                            np.pi / 4,
                            3 * np.pi / 4, 
                            5 * np.pi / 8, 
                            7 * np.pi / 8
                            ],
                            symmetric=False, normed=True, levels=32)

        #characteristics.append(shannon_entropy(glcm, base=2))
        characteristics.append(graycoprops(glcm, "energy"))
        characteristics.append(graycoprops(glcm, "correlation"))
        characteristics.append(graycoprops(glcm, "dissimilarity"))
        characteristics.append(graycoprops(glcm, "homogeneity"))

        characteristics = np.concatenate(characteristics, axis=None)

        return characteristics

    def __feed_classifier(self, x_axis, y_axis):
        self.new_information.emit("Alimentando classificador com dados", )
        self.new_status.emit("Treinando classificador")
        self.__model.fit(x_axis, y_axis)

        # Recupera todas as imagens dentro do diretorio informado

    # aplica equalização do histograma em uma imagem binaria de forma global
    def equalization_bin(self, image):
        return cv.equalizeHist(image)

    def mirror_horizontally(self, image):
        img_flip_h = cv.flip(image, 1)
        return img_flip_h

    def quantization(self,image, k):

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)/255

        # quantize and convert back to range 0 to 255 as 8-bits
        result = 255*np.floor(gray*k+0.5)/k
        result = result.clip(0,255).astype(np.uint8)
    
    def preprocess_img(self, image):
        enhancer_sharpness = ImageEnhance.Sharpness(image)
        image = enhancer_sharpness.enhance(1.6)

        enhancer_contrast = ImageEnhance.Contrast(image)
        image = enhancer_contrast.enhance(2)
        
        enhancer_brightness = ImageEnhance.Brightness(image)
        image = enhancer_brightness.enhance(1.2)

        return image
    
    # Função para ler um dado diretorio de imagens e gerar novas images quando foi marcado para
    def read_image_and_process(self, path, mirrored=False, equalization=False):
        texture = []
        kl = []
        root_image_folder = os_utils.get_all_subfolder(path)

        for i, folder in enumerate(root_image_folder):
          #  images = self.__extract_images_from_folder( os.path.join(path, folder))
            images_paths = os_utils.get_images_from_path(os.path.join(path, folder))
            
            j = 0
            l = len(images_paths)
            
            for file in images_paths:
                image = Image.open(file)

                image = self.preprocess_img(image)
   
                j += 1
                self.new_information.emit(
                    f'kl #{i}| Imagem #{j:04}/{l:04} ({round((j/l)*100,1):02}%)')

                if equalization:
                    equalization_img =  ImageOps.equalize(image)
                    
                    # Definido o numero de tons de cinza 
                    equalization_img = equalization_img.quantize(32)

                    equali_desc = self.__extract_image_texture_descriptors(
                        equalization_img)
                    texture.append(equali_desc)
                    kl.append(self.__kl_classes.index(self.__kl_classes[i]))

                if mirrored:
                    mirror_img = ImageOps.mirror(image)
                    mirror_img = mirror_img.quantize(32)
                    mirror_desc = self.__extract_image_texture_descriptors(mirror_img)
                    texture.append(mirror_desc)
                    kl.append(self.__kl_classes.index(
                        self.__kl_classes[i]))

                image = image.quantize(32)
                descriptor = self.__extract_image_texture_descriptors(image)
                texture.append(descriptor)
                kl.append(self.__kl_classes.index(self.__kl_classes[i]))

        return (texture, kl)

    # Função para chamar o treino e o teste de forma automatica
    def setup(self, root_path, mirrored=False, equalization=False):
      
        if str.__contains__(root_path,'.data'):
            root_path = os.path.join( os.getcwd(),'.cache',root_path)
            self.new_information.emit(f'Load model from:\n{root_path}')
            self.load_Model(root_path)
            data_frame_test = self.load_dataFrame(root_path)
            self.test_classifier_data_frame(data_frame_test)
        else:
            self.train_classifier(
                root_path, mirrored=mirrored, equalization=equalization)

            self.test_classifier(root_path, subfloder='test')

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

    def test_classifier(self, root_path, subfloder):

        training_timer = Stopwatch()
        training_timer.start()

        texture_test = []
        kl_test = []
        test_path = os.path.join(root_path, subfloder)

        self.new_status.emit("Lendo e processando imagens")
        texture_test, kl_test = self.read_image_and_process(test_path)

        data_frame_test = pd.DataFrame(np.array(texture_test))
        data_frame_test['Target'] = np.array(kl_test)

        x_axis_test = data_frame_test.iloc[:, :-1]
        y_axis_test = data_frame_test.iloc[:, -1]

        y_pred = self.__model.predict(x_axis_test)

        training_timer.stop()

        self.__accuracy = "{:.2f}".format(
            accuracy_score(y_axis_test, y_pred) * 100)

        self.__specificity = (100 - float(self.__accuracy))/ len(texture_test) 

        self.__f1_score = "{:.2f}".format(f1_score(y_axis_test, y_pred, average='weighted'))

        self.__precision_score = "{:.2f}".format(precision_score(y_axis_test, y_pred,average='weighted'))

        self.__sensitivity = "{:.2f}".format(recall_score(y_axis_test, y_pred,average='weighted'))

        self.__confusion_matrix = confusion_matrix(y_axis_test, y_pred)

        self.new_information.emit(f'Treinamento completo')
        uc.save_images_data_frame( data=data_frame_test,
                                   path=root_path, suffix=subfloder)
        self.new_information.emit(f'Acurácia: {self.__accuracy}')
        self.new_information.emit(f'Especificidade: {self.__specificity}')
        self.new_information.emit(f'Matriz de confusão: {self.__confusion_matrix}')
        self.new_information.emit(f'Tempo de treinamento: {training_timer.get_time()}s')
        self.new_information.emit(f'Melhores Parametros = {self.__model.best_params_}')

        self.__times['time_to_train'] = training_timer.get_time()

        self.new_status.emit("Treinamento completo")

    def train_classifier(self, root_path,  mirrored=False, equalization=False):

        training_timer = Stopwatch()
        training_timer.start()

        # iniciado valores
        texture_train = []
        kl_train = []

        train_path = os.path.join(root_path, 'train')

        texture_train, kl_train = self.read_image_and_process(
            train_path, mirrored, equalization)

        data_frame_train = pd.DataFrame(np.array(texture_train))
        data_frame_train['Target'] = np.array(kl_train)

        x_axis_train = data_frame_train.iloc[:, :-1]
        y_axis_train = data_frame_train.iloc[:, -1]

        self.__model = GridSearchCV(
            self.__classifier, self.__param_grid, n_jobs=10)

        self.__feed_classifier(x_axis_train, y_axis_train)

        training_timer.stop()
        uc.save(model_name=ModelType.SVM, data=self.__model, type_ouput='data',
                path=root_path, suffix='train', equalization=equalization, mirrored=mirrored)
        uc.save(model_name=ModelType.SVM, data=self.__times, type_ouput='txt',
                path=root_path, suffix='train', equalization=equalization, mirrored=mirrored)

        training_timer.stop()
        self.__times['training'] = training_timer.get_time()

    # Dada uma imagem é testado com o modelo treinado
    def predict_kl_image(self, image):
        prediction_timer = Stopwatch()
        prediction_timer.start()

        new_image = Image.open(image)
        new_image = new_image.quantize(32)

        descriptors = self.__extract_image_texture_descriptors(new_image)
        kl_class = self.__model.predict([descriptors])[0]

        prediction_timer.stop()
        self.__times['prediction'] = prediction_timer.get_time()

        return self.__kl_classes[kl_class]

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
