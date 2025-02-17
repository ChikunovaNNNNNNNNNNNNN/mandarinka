import sys
import cv2
import pygame
import numpy as np
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QUrl, QTimer
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog


class DetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Player")
        self.setWindowIcon(QIcon("й.jpg"))

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.5)  # Устанавливаем громкость
        url = QUrl.fromLocalFile("pd.mp3")  # Укажите путь к вашему музыкальному файлу
        self.player.setSource(url)
        self.player.mediaStatusChanged.connect(self.loop_music)
        self.player.play()

        self.setStyleSheet("""
                            QWidget {
                                background-color: #333333;
                                color: #ffffff;
                            }
                            QPushButton {
                                background-color: #2a4238;
                                color: #ffffff;
                                border: none;
                                padding: 5px;
                            }
                            QPushButton:hover {
                                background-color: #7b917b;
                            }
                        """)
        self.setWindowTitle("Обнаружение облаков и пожаров")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.load_button = QPushButton('Загрузить изображение', self)
        self.detect_clouds_button = QPushButton('Выделить облака', self)
        self.detect_fire_button = QPushButton('Выделить пожары', self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.detect_clouds_button)
        layout.addWidget(self.detect_fire_button)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # Connect buttons to functions
        self.load_button.clicked.connect(self.load_image)
        self.detect_clouds_button.clicked.connect(self.detect_clouds)
        self.detect_fire_button.clicked.connect(self.detect_fire)

        self.image = None
        self.result_image = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Открыть изображение', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)

    def detect_clouds(self):
        if self.image is not None:
            # Преобразование изображения в серый цвет
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Применяем пороговую обработку для выделения облаков (светлые области)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Найдем контуры облаков
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Рисуем контуры облаков на исходном изображении
            self.result_image = self.image.copy()
            cv2.drawContours(self.result_image, contours, -1, (0, 255, 0), 2)

    def loop_music(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0)  # Перематываем в начало
            self.player.play()

    def detect_fire(self):
        if self.image is not None:
            # Преобразуем изображение в формат HSV (для удобства выделения красных и оранжевых оттенков)
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

            # Определяем диапазоны для красного и оранжевого цветов (пожары)
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([25, 255, 255])

            # Маски для красного и оранжевого цветов
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

            # Объединяем маски для красного и оранжевого цветов
            fire_mask = cv2.bitwise_or(red_mask, orange_mask)

            # Найдем контуры пожара
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Рисуем контуры пожара на исходном изображении
            self.result_image = self.image.copy()
            cv2.drawContours(self.result_image, contours, -1, (0, 0, 255), 2)

            self.display_image(self.result_image)

    def display_image(self, image):
        """Конвертируем изображение OpenCV в формат, который может быть отображен в PyQt"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))


def main():
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
