# Condigo reaproveitado do semestre anteiror 
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSlot as Slot


class ViewConfusionMatriz(QWidget):

    def __init__(self, parent=None) -> None:
            super().__init__(parent)

            self.layout = QVBoxLayout()
            self.setLayout(self.layout)

    def remove_children(self):
        for i in reversed(range(self.layout.count())):
            self.imageInfoColumn.itemAt(i).widget().setParent(None)

    # @Slot
    def plot_confusion_matrix(self,confusion_matrix,sensibility,specificity):

        confusion_matrix = confusion_matrix.replace(" ", ",")
        confusion_matrix = confusion_matrix.replace(",,", ",")
        confusion_matrix = confusion_matrix.replace("[,", "[")
        confusion_matrix = ast.literal_eval(confusion_matrix)

        static_canvas = FigureCanvas(Figure(figsize=(5, 3),tight_layout=True))

        self.remove_children()

        self.layout.addWidget(NavigationToolbar(static_canvas, self))
        self.layout.addWidget(static_canvas)

        self._static_ax = static_canvas.figure.subplots()
   
        self._static_ax.matshow(confusion_matrix, cmap=plt.cm.Oranges)

        self._static_ax.set_ylabel("Actual")
        self._static_ax.set_xlabel("Predicted")

        self._static_ax.set_xticklabels(("0","1","2","3","4"))
        self._static_ax.set_yticklabels(("0","1","2","3","4"))

        specificity = "{:.2f}".format(float(specificity))

        self._static_ax.title.set_text("Sensibility: " + str(sensibility) + ", Specificity: " + str(specificity))
 