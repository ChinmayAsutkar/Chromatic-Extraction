import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class BackgroundEstimatorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Robust Background Estimation')
        self.setGeometry(100, 100, 1800, 1000)

        main_layout = QVBoxLayout()

        grid_layout = QGridLayout()

        self.image_label = QLabel('Original Image')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")

        self.background_label = QLabel('Background')
        self.background_label.setAlignment(Qt.AlignCenter)
        self.background_label.setStyleSheet("border: 1px solid black;")

        self.mask_label = QLabel('Mask')
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setStyleSheet("border: 1px solid black;")

        self.foreground_label = QLabel('Foreground')
        self.foreground_label.setAlignment(Qt.AlignCenter)
        self.foreground_label.setStyleSheet("border: 1px solid black;")

        grid_layout.addWidget(self.image_label, 0, 0)
        grid_layout.addWidget(self.background_label, 0, 1)
        grid_layout.addWidget(self.mask_label, 1, 0)
        grid_layout.addWidget(self.foreground_label, 1, 1)

        # Load button
        self.load_button = QPushButton('Select Image')
        self.load_button.clicked.connect(self.load_image)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.load_button)
        button_layout.addStretch()

        main_layout.addLayout(grid_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                return
            background, foreground, mask = estimate_background(image)

            self.display_image(image, self.image_label, fit=True)
            self.display_image(background, self.background_label, fit=True)
            self.display_image(foreground, self.foreground_label, fit=True)
            self.display_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), self.mask_label, fit=True)

    def display_image(self, image, label, fit=False):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if fit:
            pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)


def estimate_background(image, clusters=3):
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    pixels = image_lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.flatten()
    counts = np.bincount(labels)
    background_label = np.argmax(counts)
    background_color_lab = centers[background_label]

    background_image_lab = np.full_like(image_lab, background_color_lab)
    background_image = cv2.cvtColor(background_image_lab, cv2.COLOR_LAB2BGR)

    # Convert to grayscale for difference
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.absdiff(image_gray, background_gray)

    # Adaptive threshold for better ink separation
    mask_foreground = cv2.adaptiveThreshold(
        diff_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 35, -5
    )

    # Invert to highlight ink (foreground as white)
    mask_foreground = cv2.bitwise_not(mask_foreground)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    foreground = cv2.bitwise_and(image, image, mask=mask_clean)

    return background_image, foreground, mask_clean


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BackgroundEstimatorUI()
    window.show()
    sys.exit(app.exec_())
