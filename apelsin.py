import sys
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog

class CloudDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Обнаружение облаков")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.load_button = QPushButton('Загрузить изображение', self)
        self.detect_button = QPushButton('Выделить облака', self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # Connect buttons to functions
        self.load_button.clicked.connect(self.load_image)
        self.detect_button.clicked.connect(self.detect_clouds)

        self.image = None
        self.clouds_image = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Открыть изображение', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)

    def detect_clouds(self):
        if self.image is not None:
            # Преобразование изображения в серый цвет
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Применение пороговой обработки для выделения облаков
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Найдем контуры (облака)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Рисуем контуры на исходном изображении
            self.clouds_image = self.image.copy()
            cv2.drawContours(self.clouds_image, contours, -1, (0, 255, 0), 2)

            self.display_image(self.clouds_image)

    def display_image(self, image):
        """Конвертируем изображение OpenCV в формат, который может быть отображен в PyQt"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CloudDetectionApp()
    window.show()
    sys.exit(app.exec())
