# This Python file uses the following encoding: utf-8
from PyQt6.QtWidgets import QWidget,QVBoxLayout
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Classe reaproveitada do semestre anterio HOMENIQUE VIEIRA MARTINS

class ImageConfusionMatrix(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def remove_children(self):
        for i in reversed(range(self.layout.count())):
            self.imageInfoColumn.itemAt(i).widget().setParent(None)

    #@Slot
    def on_training_completed(self,confusion_matrix, metrics):

        static_canvas = FigureCanvas(Figure(figsize=(5, 3),tight_layout=True))

        self.remove_children()

        self.layout.addWidget(NavigationToolbar(static_canvas, self))
        self.layout.addWidget(static_canvas)

        self._static_ax = static_canvas.figure.subplots()

        self._static_ax.matshow(confusion_matrix, cmap=plt.cm.Oranges)

        self._static_ax.set_ylabel("Classe Real")
        self._static_ax.set_xlabel("Classe Prevista")

        self._static_ax.set_xticklabels(("0","0","1","2","3","4"))
        self._static_ax.set_yticklabels(("0","0","1","2","3","4"))

        self._static_ax.title.set_text(f"Matriz de Confusão")
