import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QFileDialog, QComboBox, QMainWindow, QScrollArea,
                           QSlider, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont, QKeyEvent

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selective Blue Shade Extractor")
        self.image = None
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 1800, 1200)
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
            QScrollArea {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
            }
            QSpinBox {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 2px;
                min-width: 60px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #4CAF50;
                border: none;
                border-radius: 2px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #45a049;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)

        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        # Create container widgets for better organization
        hue_tolerance_container = QWidget()
        hue_tolerance_layout = QVBoxLayout(hue_tolerance_container)
        self.intensity_label = QLabel("Hue Tolerance: 10")
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(50)
        self.intensity_slider.setValue(10)
        self.intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.intensity_slider.setTickInterval(5)
        self.intensity_slider.valueChanged.connect(self.update_hue_tolerance_label)
        self.intensity_slider.setFocusPolicy(Qt.StrongFocus)
        self.intensity_slider.setToolTip("Use arrow keys or type numbers to adjust\nLeft/Right arrows: Fine adjustment\nUp/Down arrows: Coarse adjustment")
        
        # Add spinbox for direct value input
        self.intensity_spinbox = QSpinBox()
        self.intensity_spinbox.setRange(0, 50)
        self.intensity_spinbox.setValue(10)
        self.intensity_spinbox.valueChanged.connect(self.intensity_slider.setValue)
        self.intensity_slider.valueChanged.connect(self.intensity_spinbox.setValue)
        
        hue_tolerance_layout.addWidget(self.intensity_label)
        hue_tolerance_layout.addWidget(self.intensity_slider)
        hue_tolerance_layout.addWidget(self.intensity_spinbox)

        target_hue_container = QWidget()
        target_hue_layout = QVBoxLayout(target_hue_container)
        self.hue_label = QLabel("Target Hue: 110")
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setMinimum(0)
        self.hue_slider.setMaximum(179)
        self.hue_slider.setValue(110)
        self.hue_slider.setTickPosition(QSlider.TicksBelow)
        self.hue_slider.setTickInterval(10)
        self.hue_slider.valueChanged.connect(self.update_target_hue_label)
        self.hue_slider.setFocusPolicy(Qt.StrongFocus)
        self.hue_slider.setToolTip("Use arrow keys or type numbers to adjust\nLeft/Right arrows: Fine adjustment\nUp/Down arrows: Coarse adjustment")
        
        # Add spinbox for direct value input
        self.hue_spinbox = QSpinBox()
        self.hue_spinbox.setRange(0, 179)
        self.hue_spinbox.setValue(110)
        self.hue_spinbox.valueChanged.connect(self.hue_slider.setValue)
        self.hue_slider.valueChanged.connect(self.hue_spinbox.setValue)
        
        target_hue_layout.addWidget(self.hue_label)
        target_hue_layout.addWidget(self.hue_slider)
        target_hue_layout.addWidget(self.hue_spinbox)

        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(hue_tolerance_container)
        control_layout.addWidget(target_hue_container)
        control_layout.addStretch()

        image_layout = QVBoxLayout()
        images_horizontal = QHBoxLayout()

        self.scroll_original = QScrollArea()
        self.scroll_grid = QScrollArea()
        self.scroll_result = QScrollArea()

        self.label_original = QLabel("Original Image")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setMinimumSize(500, 500)
        self.label_original.setStyleSheet("border: 1px solid #ddd;")

        self.label_grid = QLabel("Grid Image")
        self.label_grid.setAlignment(Qt.AlignCenter)
        self.label_grid.setMinimumSize(500, 500)
        self.label_grid.setStyleSheet("border: 1px solid #ddd;")

        self.label_result = QLabel("Extracted Result")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setMinimumSize(500, 500)
        self.label_result.setStyleSheet("border: 1px solid #ddd;")

        self.scroll_original.setWidget(self.label_original)
        self.scroll_grid.setWidget(self.label_grid)
        self.scroll_result.setWidget(self.label_result)

        images_horizontal.addWidget(self.scroll_original)
        images_horizontal.addWidget(self.scroll_grid)
        images_horizontal.addWidget(self.scroll_result)

        self.figure = plt.figure(figsize=(8, 2))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)

        image_layout.addLayout(images_horizontal)
        image_layout.addWidget(self.canvas)

        main_layout.addLayout(control_layout)
        main_layout.addLayout(image_layout)

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

        if len(self.image.shape) == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        elif self.image.shape[2] == 4:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGR)

        original_img = self.image.copy()
        grid_img = self.image.copy()
        h, w, _ = self.image.shape
        grid_h, grid_w = h // 100, w // 100

        hue_tolerance = self.intensity_slider.value()
        target_hue = self.hue_slider.value()

        global_mask = np.zeros((h, w), dtype=np.uint8)
        hue_values = []

        for i in range(100):
            for j in range(100):
                y1, y2 = i * grid_h, min((i + 1) * grid_h, h)
                x1, x2 = j * grid_w, min((j + 1) * grid_w, w)
                grid = self.image[y1:y2, x1:x2]

                try:
                    hsv = cv2.cvtColor(grid, cv2.COLOR_BGR2HSV)
                except cv2.error:
                    continue

                lower_hue = max(target_hue - hue_tolerance, 0)
                upper_hue = min(target_hue + hue_tolerance, 179)

                mask = cv2.inRange(hsv, (lower_hue, 50, 50), (upper_hue, 255, 255))
                global_mask[y1:y2, x1:x2] = cv2.bitwise_or(global_mask[y1:y2, x1:x2], mask)

                hue_values.append(target_hue)
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        white_bg = np.full_like(self.image, 255)
        result = np.where(global_mask[..., np.newaxis] == 255, self.image, white_bg)

        self.display_image(self.label_original, original_img)
        self.display_image(self.label_grid, grid_img)
        self.display_image(self.label_result, result)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist(hue_values, bins=180, range=(0, 180), color='blue', alpha=0.7)
        ax.set_xlabel('Hue Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Selected Hue across Grid Cells')
        self.canvas.draw()

    def display_image(self, label, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def update_hue_tolerance_label(self):
        value = self.intensity_slider.value()
        self.intensity_label.setText(f"Hue Tolerance: {value}")
        self.update_result()

    def update_target_hue_label(self):
        value = self.hue_slider.value()
        self.hue_label.setText(f"Target Hue: {value}")
        self.update_result()

    def keyPressEvent(self, event: QKeyEvent):
        focused_widget = self.focusWidget()
        
        if isinstance(focused_widget, QSlider):
            if event.key() == Qt.Key_Left:
                focused_widget.setValue(focused_widget.value() - 1)
            elif event.key() == Qt.Key_Right:
                focused_widget.setValue(focused_widget.value() + 1)
            elif event.key() == Qt.Key_Up:
                focused_widget.setValue(focused_widget.value() + 5)
            elif event.key() == Qt.Key_Down:
                focused_widget.setValue(focused_widget.value() - 5)
            elif event.key() >= Qt.Key_0 and event.key() <= Qt.Key_9:
                # Handle direct number input
                current_value = focused_widget.value()
                new_value = int(str(current_value) + chr(event.key()))
                if new_value <= focused_widget.maximum():
                    focused_widget.setValue(new_value)
        else:
            super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    processor = ImageProcessor()
    processor.show()
    sys.exit(app.exec_())