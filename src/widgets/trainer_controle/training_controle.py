import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

from skimage.io import imread


class TrainingControle:
    def __init__(self,parent= None) -> None:
        super().__init__(parent)
     



    def __extract_images_from_folder(self,image_folder):

        images = []

        images_paths = UtilsOs.get_images_from_path(image_folder)

        for file in images_paths:
            image = imread(file,as_gray=True)
            images.append(image)

        return images
    
    def __extract_image_texture_descriptors(self,image):

        characteristics = []

        glcm = graycomatrix(image, 
                            [1,2,4,8,16],
                            [0, np.pi / 2, np.pi / 4, np.pi / 8, 3 * np.pi / 4, 5 * np.pi / 8, 7 * np.pi/8],
                            symmetric=False, 
                            normed=True)

        characteristics.append(shannon_entropy(glcm,base=2))
        characteristics.append(graycoprops(glcm,"energy"))
        characteristics.append(graycoprops(glcm,"correlation"))
        characteristics.append(graycoprops(glcm,"dissimilarity"))
        characteristics.append(graycoprops(glcm,"homogeneity"))

        characteristics = np.concatenate(characteristics,axis=None)

        return characteristics 
