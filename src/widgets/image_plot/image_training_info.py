
from PyQt6.QtWidgets import QWidget,QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QHeaderView
from PyQt6.QtCore import Qt

class ImageTrainingInfo(QWidget):

    def remove_children(self):
        for i in reversed(range(self.layout.count())):
            self.imageInfoColumn.itemAt(i).widget().setParent(None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vbox = QVBoxLayout(self)

        self.column = 5
        self.row = 2

        self.table_widget = QTableWidget()
        self.horizontal_header = QHeaderView(Qt.Orientation.Horizontal)
        self.vertical_header = QHeaderView(Qt.Orientation.Vertical)
        self.table_widget.setHorizontalHeader(self.horizontal_header)
        self.table_widget.setVerticalHeader(self.vertical_header)
        self.horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.horizontal_header.setVisible(False)
        self.vertical_header.setVisible(False)

        self.table_widget.setRowCount(5)
        self.table_widget.setColumnCount(2)

        self.vbox.addWidget(self.table_widget)

    def on_training_completed(self,_, metrics):

        self.table_widget.clearContents()    

        for index, (key, value) in enumerate(metrics.items()):
            self.table_widget.setItem(index,0,QTableWidgetItem(key))
            self.table_widget.setItem(index,1,QTableWidgetItem(str(value)))


