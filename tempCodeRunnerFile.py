import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QFileDialog, QLabel, QHBoxLayout, QSlider,
                           QSpinBox, QGridLayout, QScrollArea, QSizePolicy,
                           QDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

class ZoomWindow(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zoomed Image")
        self.setModal(True)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Scale image to a larger size (e.g., 600x600)
        scaled_pixmap = image.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        
        # Add label to layout
        layout.addWidget(self.image_label)
        
        # Set window size
        self.resize(650, 650)

class ColorExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.value = 128  # Default value (middle brightness)
        self.saturation = 50  # Default saturation threshold
        self.captured_images = []  # List to store captured images
        self.max_captures = 5
        self.captured_containers = []  # List to store container widgets
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Color Extractor")
        self.setGeometry(100, 100, 1600, 900)  # Increased height
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

        # Saturation controls
        saturation_container = QWidget()
        saturation_layout = QVBoxLayout(saturation_container)
        self.saturation_label = QLabel("Saturation Threshold: 50")
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setMinimum(0)
        self.saturation_slider.setMaximum(255)
        self.saturation_slider.setValue(50)
        self.saturation_slider.valueChanged.connect(self.update_saturation_label)
        
        self.saturation_spinbox = QSpinBox()
        self.saturation_spinbox.setRange(0, 255)
        self.saturation_spinbox.setValue(50)
        self.saturation_spinbox.valueChanged.connect(self.saturation_slider.setValue)
        self.saturation_slider.valueChanged.connect(self.saturation_spinbox.setValue)
        
        saturation_layout.addWidget(self.saturation_label)
        saturation_layout.addWidget(self.saturation_slider)
        saturation_layout.addWidget(self.saturation_spinbox)
        control_layout.addWidget(saturation_container)

        # Value controls
        value_container = QWidget()
        value_layout = QVBoxLayout(value_container)
        self.value_label = QLabel("Value: 128")
        self.value_slider = QSlider(Qt.Horizontal)
        self.value_slider.setMinimum(0)
        self.value_slider.setMaximum(255)
        self.value_slider.setValue(128)
        self.value_slider.valueChanged.connect(self.update_value_label)
        
        self.value_spinbox = QSpinBox()
        self.value_spinbox.setRange(0, 255)
        self.value_spinbox.setValue(128)
        self.value_spinbox.valueChanged.connect(self.value_slider.setValue)
        self.value_slider.valueChanged.connect(self.value_spinbox.setValue)
        
        value_layout.addWidget(self.value_label)
        value_layout.addWidget(self.value_slider)
        value_layout.addWidget(self.value_spinbox)
        control_layout.addWidget(value_container)

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

        # Capture button
        capture_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Result")
        self.capture_btn.clicked.connect(self.capture_result)
        self.capture_btn.setEnabled(False)  # Disabled until an image is loaded
        capture_layout.addWidget(self.capture_btn)
        capture_layout.addStretch()
        main_layout.addLayout(capture_layout)

        # Create a horizontal layout for captured images and histogram
        bottom_layout = QHBoxLayout()

        # Captured images scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(250)  # Fixed height for scroll area
        self.scroll_widget = QWidget()  # Store as instance variable
        self.captured_grid = QGridLayout(self.scroll_widget)
        self.captured_grid.setSpacing(10)
        scroll_area.setWidget(self.scroll_widget)
        bottom_layout.addWidget(scroll_area, 1)  # Takes 1/3 of the space

        # Histogram with fixed size
        histogram_container = QWidget()
        histogram_layout = QVBoxLayout(histogram_container)
        self.figure = plt.figure(figsize=(6, 2))  # Reduced figure size
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedHeight(200)  # Fixed height for histogram
        histogram_layout.addWidget(self.canvas)
        bottom_layout.addWidget(histogram_container, 2)  # Takes 2/3 of the space

        main_layout.addLayout(bottom_layout)

        self.captured_labels = []  # List to store labels for captured images
        self.delete_buttons = []   # List to store delete buttons

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                self.image = cv2.imread(path)
                if self.image is None:
                    raise Exception("Failed to load image")
                self.capture_btn.setEnabled(True)  # Enable capture button when image is loaded
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
        saturation_threshold = self.saturation
        value_threshold = self.value

        # Create mask with saturation and value thresholds
        lower_hue = max(target_hue - hue_tolerance, 0)
        upper_hue = min(target_hue + hue_tolerance, 179)
        mask = cv2.inRange(hsv, 
                          (lower_hue, saturation_threshold, value_threshold),
                          (upper_hue, 255, 255))

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
        
        # Calculate area for each hue value
        total_pixels = hsv.shape[0] * hsv.shape[1]
        hue_areas = np.zeros(180)
        hue_frequencies = np.zeros(180)
        
        # For each hue value, calculate the percentage of area and frequency
        for hue in range(180):
            # Create mask for this specific hue
            mask = cv2.inRange(hsv, (hue, 50, 50), (hue, 255, 255))
            # Calculate percentage of area
            area_percentage = (np.sum(mask == 255) / total_pixels) * 100
            hue_areas[hue] = area_percentage
            # Calculate frequency (number of pixels with this hue)
            hue_frequencies[hue] = np.sum(mask == 255)
        
        # Create color array for each hue value
        colors = np.zeros((180, 3))
        for hue in range(180):
            # Convert HSV to RGB for visualization
            hsv_color = np.uint8([[[hue, 255, 255]]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0] / 255.0
            colors[hue] = rgb_color
        
        # Create filled area plot with colors
        x = np.arange(180)
        
        # Plot area coverage with logarithmic scale
        ax.fill_between(x, hue_areas, color='none', alpha=0.3)
        
        # Add colored segments with logarithmic scale
        for i in range(180):
            if hue_areas[i] > 0:  # Only plot non-zero values
                ax.fill_between([i, i+1], [0, 0], [hue_areas[i], hue_areas[i]], 
                              color=colors[i], alpha=0.7)
        
        # Add vertical lines for target hue and tolerance
        target_hue = self.hue_slider.value()
        tolerance = self.tolerance_slider.value()
        ax.axvline(x=target_hue, color='red', linestyle='--', label='Target Hue')
        ax.axvline(x=max(target_hue - tolerance, 0), color='green', linestyle=':', label='Tolerance Range')
        ax.axvline(x=min(target_hue + tolerance, 179), color='green', linestyle=':')
        
        # Calculate and display the total area coverage within tolerance range
        total_coverage = np.sum(hue_areas[max(0, target_hue - tolerance):min(180, target_hue + tolerance + 1)])
        total_frequency = np.sum(hue_frequencies[max(0, target_hue - tolerance):min(180, target_hue + tolerance + 1)])
        
        # Add text box with both area coverage and frequency information
        textstr = f'Total Coverage: {total_coverage:.1f}%\\nTotal Pixels: {total_frequency:,}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set logarithmic scale for y-axis
        ax.set_yscale('log')
        
        # Set y-axis limits to show small values better
        max_val = np.max(hue_areas[hue_areas > 0]) if np.any(hue_areas > 0) else 1
        min_val = np.min(hue_areas[hue_areas > 0]) if np.any(hue_areas > 0) else 0.001
        ax.set_ylim(bottom=min_val/2, top=max_val*2)
        
        ax.set_xlabel('Hue Value')
        ax.set_ylabel('Area Coverage (%) - Log Scale')
        ax.set_title('Hue Distribution by Area Coverage')
        ax.legend()
        
        # Add grid for better readability
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
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

    def update_saturation_label(self):
        value = self.saturation_slider.value()
        self.saturation_label.setText(f"Saturation Threshold: {value}")
        self.saturation = value
        self.update_result()

    def update_value_label(self):
        value = self.value_slider.value()
        self.value_label.setText(f"Value: {value}")
        self.value = value
        self.update_result()

    def capture_result(self):
        if len(self.captured_images) >= self.max_captures:
            return

        # Get the current result image
        result_pixmap = self.result_label.pixmap()
        if result_pixmap:
            # Convert QPixmap to QImage
            result_image = result_pixmap.toImage()
            
            # Store the captured image
            self.captured_images.append(result_image)
            
            # Create a container widget for image and delete button
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(5)
            
            # Create a new label for the captured image
            label = QLabel()
            label.setFixedSize(180, 180)  # Slightly smaller size
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid #ddd;")
            label.setCursor(Qt.PointingHandCursor)  # Change cursor to hand when hovering
            label.mousePressEvent = lambda event, idx=len(self.captured_images)-1: self.show_zoom_window(idx)
            
            # Create delete button
            delete_btn = QPushButton("Delete")
            delete_btn.setFixedWidth(80)
            delete_btn.clicked.connect(lambda checked, idx=len(self.captured_images)-1: self.delete_capture(idx))
            
            # Add widgets to container
            container_layout.addWidget(label)
            container_layout.addWidget(delete_btn, alignment=Qt.AlignCenter)
            
            # Add to grid
            row = len(self.captured_images) - 1
            self.captured_grid.addWidget(container, row // 2, row % 2)  # 2 columns layout
            
            # Store references
            self.captured_labels.append(label)
            self.delete_buttons.append(delete_btn)
            self.captured_containers.append(container)
            
            # Display the captured image
            scaled_pixmap = result_pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            
            # Disable capture button if max captures reached
            if len(self.captured_images) >= self.max_captures:
                self.capture_btn.setEnabled(False)

    def delete_capture(self, index):
        if index < 0 or index >= len(self.captured_images):
            return

        # Remove the image and its UI elements
        self.captured_images.pop(index)
        
        # Remove the container from the grid
        container = self.captured_containers[index]
        self.captured_grid.removeWidget(container)
        container.deleteLater()
        
        # Remove from our lists
        self.captured_labels.pop(index)
        self.delete_buttons.pop(index)
        self.captured_containers.pop(index)
        
        # Reorganize remaining captures
        for i in range(len(self.captured_images)):
            container = self.captured_containers[i]
            self.captured_grid.addWidget(container, i // 2, i % 2)
        
        # Re-enable capture button if below max
        if len(self.captured_images) < self.max_captures:
            self.capture_btn.setEnabled(True)

    def show_zoom_window(self, index):
        if 0 <= index < len(self.captured_images):
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(self.captured_images[index])
            # Create and show zoom window
            zoom_window = ZoomWindow(pixmap, self)
            zoom_window.exec_()

def main():
    app = QApplication(sys.argv)
    ex = ColorExtractor()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 