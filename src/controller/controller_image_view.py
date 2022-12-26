from PyQt6.QtWidgets import QFileDialog
import cv2 as cv
import os

class ControllerImageView:

    def __init__(self,handler) -> None:
            self.handler = handler
            self.image = None
            self.image_path = None
            self.image_was_loaded = False

    # Método para abrir um arquivo (.png ou .jpg)
    def open_file (self)->str:
        self.image_path = QFileDialog.getOpenFileName(self.handler,
            str("Open Image"), os.path.expanduser('~'), filter="PNG(*.png);;JPG(*.jpg)")[0]
        return self.image_path

    def load_image(self, image_path):
        return cv.imread(image_path,0)

    # Atualiza a imagem no Qt
    def set_image(self, image):
        dimensions = image.shape

        height = dimensions[0]
        width = dimensions[1]
        
        if height < 250 and width < 250:
            width = int(width * 2)
            height = int(height * 2)
            dim = (width, height)
            newimage = cv.resize(image, dim, interpolation = cv.INTER_AREA)
         
            self.handler.set_image_OpenCV(newimage)
        else :
            self.handler.set_image_OpenCV(image)

    # Abre o arquivo de imagem e atualiza no Qt
    def open_image (self)->str:
        path_image = self.open_file()
        
        if not path_image:
            return
        
        self.image = self.load_image(path_image)
        self.image_was_loaded = True

        self.set_image(self.image)

    # Método para encontrar uma região em uma imagem
    # Essa região é encontrada baseada em uma outra imagem salva
    # Utilizamos o método matchTemplate do opencv
    def find_sub_image(self)->str:
        if not self.image_was_loaded:
            return
        points = []

        # Abre o arquivo da região a ser encontrada
        path_sub_image = self.open_file()

        # if self.image == None and path_sub_image == None: 
        if path_sub_image == '': 
            return 

        points = self.detected_sub_image(path_sub_image,'cv.TM_CCOEFF_NORMED')

        # Desenha o retangulo na região encontrada na imagem atual
        clone = self.image.copy()


        cv.rectangle(clone, points[0], points[1], 0, 2)
        self.set_image(clone)


    # Função de detectar a região na imagem
    def detected_sub_image(self, path_sub_image, method)->list:
        image = self.image.copy()
        sub_image = cv.imread(path_sub_image, 0) 
        points = [] 
        width, height = sub_image.shape[::-1]

        # Template Matching
        mt = cv.matchTemplate(image,sub_image, eval(method))
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(mt)


        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else: 
            top_left = max_loc


        bottom_right = (top_left[0] + width, top_left[1]+ height)
        points.append(top_left)
        points.append(bottom_right)
    
        return points

    def save_image(self): 
        if not self.image_was_loaded:  
            return

        file = QFileDialog.getSaveFileName(self.handler,
            str("Save Image File"), os.path.expanduser('~'), filter="JPG(*.jpg);;PNG(*.png)")[0]

        if not file:
            return

        cv.imwrite(file,self.image)

    # Método para fazer um recorte na imagem
    # Utilizando o selectROI do opencv
    def crop_image(self):
        roi = cv.selectROI('crop', self.image, showCrosshair=False)
        cv.moveWindow('crop', 200,200)
        cv.destroyWindow('crop')

        print(roi)

        # Se não cancelou o recorte, atualiza a imagem
        if (roi[0] != 0 and roi[1] != 0 and roi[2] != 0 and roi[3] != 0):
            img_cropped = self.image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            self.image = img_cropped
            self.set_image(self.image)