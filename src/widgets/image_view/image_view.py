from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap,QImage
from PyQt6.QtCore import pyqtSlot as Slot
from controller.controller_image_view import ControllerImageView 
import cv2 as cv

class ImageView(QLabel):
    def __init__(self,parent=None) -> None:
        super().__init__(parent)
        self.imageControle = ControllerImageView(self)

    def set_image(self,file):
        super().setPixmap(QPixmap(file))

    def set_image_OpenCV(self,image):
        frame = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format.Format_RGB888)
        super().setPixmap(QPixmap.fromImage(image))