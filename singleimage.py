import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QFileDialog, QLabel, QHBoxLayout, QSlider,
                           QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

class ColorExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Color Extractor")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
                min-width: 150px;
            }
            QPushButton:hover { 
                background-color: #45a049; 
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #f0f0f0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QLabel { 
                font-size: 14px;
                color: #ffffff;
                font-weight: bold;
            }
            QSpinBox {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 2px;
                min-width: 60px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control layout
        control_layout = QHBoxLayout()
        
        # Load button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)

        # Hue tolerance controls
        hue_tolerance_container = QWidget()
        hue_tolerance_layout = QVBoxLayout(hue_tolerance_container)
        self.tolerance_label = QLabel("Hue Tolerance: 10")
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(0)
        self.tolerance_slider.setMaximum(50)
        self.tolerance_slider.setValue(10)
        self.tolerance_slider.valueChanged.connect(self.update_tolerance_label)
        
        self.tolerance_spinbox = QSpinBox()
        self.tolerance_spinbox.setRange(0, 50)
        self.tolerance_spinbox.setValue(10)
        self.tolerance_spinbox.valueChanged.connect(self.tolerance_slider.setValue)
        self.tolerance_slider.valueChanged.connect(self.tolerance_spinbox.setValue)
        
        hue_tolerance_layout.addWidget(self.tolerance_label)
        hue_tolerance_layout.addWidget(self.tolerance_slider)
        hue_tolerance_layout.addWidget(self.tolerance_spinbox)
        control_layout.addWidget(hue_tolerance_container)

        # Target hue controls
        target_hue_container = QWidget()
        target_hue_layout = QVBoxLayout(target_hue_container)
        self.hue_label = QLabel("Target Hue: 110")
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setMinimum(0)
        self.hue_slider.setMaximum(179)
        self.hue_slider.setValue(110)
        self.hue_slider.valueChanged.connect(self.update_hue_label)
        
        self.hue_spinbox = QSpinBox()
        self.hue_spinbox.setRange(0, 179)
        self.hue_spinbox.setValue(110)
        self.hue_spinbox.valueChanged.connect(self.hue_slider.setValue)
        self.hue_slider.valueChanged.connect(self.hue_spinbox.setValue)
        
        target_hue_layout.addWidget(self.hue_label)
        target_hue_layout.addWidget(self.hue_slider)
        target_hue_layout.addWidget(self.hue_spinbox)
        control_layout.addWidget(target_hue_container)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # Image display layout
        image_layout = QHBoxLayout()
        
        # Original image
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setStyleSheet("border: 1px solid #ddd;")
        image_layout.addWidget(self.original_label)

        # Result image
        self.result_label = QLabel("Extracted Result")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 400)
        self.result_label.setStyleSheet("border: 1px solid #ddd;")
        image_layout.addWidget(self.result_label)

        main_layout.addLayout(image_layout)

        # Histogram
        self.figure = plt.figure(figsize=(8, 2))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        main_layout.addWidget(self.canvas)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                self.image = cv2.imread(path)
                if self.image is None:
                    raise Exception("Failed to load image")
                self.update_result()
            except Exception as e:
                print(f"Error loading image: {str(e)}")

    def update_result(self):
        if self.image is None:
            return

        # Convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Get current values
        hue_tolerance = self.tolerance_slider.value()
        target_hue = self.hue_slider.value()

        # Create mask
        lower_hue = max(target_hue - hue_tolerance, 0)
        upper_hue = min(target_hue + hue_tolerance, 179)
        mask = cv2.inRange(hsv, (lower_hue, 50, 50), (upper_hue, 255, 255))

        # Apply mask
        white_bg = np.full_like(self.image, 255)
        result = np.where(mask[..., np.newaxis] == 255, self.image, white_bg)

        # Display images
        self.display_image(self.original_label, self.image)
        self.display_image(self.result_label, result)

        # Update histogram
        self.update_histogram(hsv)

    def update_histogram(self, hsv):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Calculate histogram for hue channel
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        ax.plot(hist, color='blue', alpha=0.7)
        
        # Add vertical lines for target hue and tolerance
        target_hue = self.hue_slider.value()
        tolerance = self.tolerance_slider.value()
        ax.axvline(x=target_hue, color='red', linestyle='--', label='Target Hue')
        ax.axvline(x=max(target_hue - tolerance, 0), color='green', linestyle=':', label='Tolerance Range')
        ax.axvline(x=min(target_hue + tolerance, 179), color='green', linestyle=':')
        
        ax.set_xlabel('Hue Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Hue Distribution')
        ax.legend()
        
        self.canvas.draw()

    def display_image(self, label, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def update_tolerance_label(self):
        value = self.tolerance_slider.value()
        self.tolerance_label.setText(f"Hue Tolerance: {value}")
        self.update_result()

    def update_hue_label(self):
        value = self.hue_slider.value()
        self.hue_label.setText(f"Target Hue: {value}")
        self.update_result()

def main():
    app = QApplication(sys.argv)
    ex = ColorExtractor()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()