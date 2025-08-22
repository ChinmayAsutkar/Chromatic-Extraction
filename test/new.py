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
        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        scaled_pixmap = image.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        layout.addWidget(self.image_label)
        self.resize(650, 650)

class ColorExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.contrast = 1.0
        self.saturation = 50
        self.captured_images = []
        self.max_captures = 10
        self.hue_segments = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Color Extractor")
        self.setGeometry(100, 100, 1600, 900)
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
            QLabel { 
                font-size: 14px;
                color: #ffffff;
                font-weight: bold;
            }
            QSlider {
                height: 20px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #4a4a4a;
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

        # Extract button
        self.extract_btn = QPushButton("Extract Layers")
        self.extract_btn.clicked.connect(self.extract_layers)
        self.extract_btn.setEnabled(False)  # Disabled until image is loaded
        control_layout.addWidget(self.extract_btn)

        # Add contrast and saturation controls
        contrast_layout = QVBoxLayout()
        contrast_label = QLabel("Contrast")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(50)
        self.contrast_slider.setMaximum(150)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_slider)
        control_layout.addLayout(contrast_layout)

        saturation_layout = QVBoxLayout()
        saturation_label = QLabel("Saturation")
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setMinimum(0)
        self.saturation_slider.setMaximum(100)
        self.saturation_slider.setValue(50)
        self.saturation_slider.valueChanged.connect(self.update_saturation)
        saturation_layout.addWidget(saturation_label)
        saturation_layout.addWidget(self.saturation_slider)
        control_layout.addLayout(saturation_layout)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # Image and histogram layout
        top_layout = QHBoxLayout()

        # Original image container
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setStyleSheet("border: 1px solid #ddd;")
        original_layout.addWidget(self.original_label)
        top_layout.addWidget(original_container)

        # Histogram container
        histogram_container = QWidget()
        histogram_layout = QVBoxLayout(histogram_container)
        self.figure = plt.figure(figsize=(8, 2))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        histogram_layout.addWidget(self.canvas)
        top_layout.addWidget(histogram_container)

        main_layout.addLayout(top_layout)

        # Scroll area for segments
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(300)  # Fixed height for scroll area
        self.scroll_widget = QWidget()
        self.segment_grid = QGridLayout(self.scroll_widget)
        self.segment_grid.setSpacing(10)
        scroll_area.setWidget(self.scroll_widget)
        main_layout.addWidget(scroll_area)

    def update_contrast(self):
        if self.image is not None:
            self.contrast = self.contrast_slider.value() / 100.0
            self.calculate_histogram()

    def update_saturation(self):
        if self.image is not None:
            self.saturation = self.saturation_slider.value()
            self.calculate_histogram()

    def find_all_local_maxima(self, data, window_size=1):
        """Return all local maxima indices in the data, including plateaus, using a small window."""
        maxima = []
        for i in range(window_size, len(data)-window_size):
            left = all(data[i] >= data[i-j] for j in range(1, window_size+1))
            right = all(data[i] >= data[i+j] for j in range(1, window_size+1))
            strictly_greater = any(data[i] > data[i-j] for j in range(1, window_size+1)) or any(data[i] > data[i+j] for j in range(1, window_size+1))
            if left and right and strictly_greater:
                maxima.append(i)
        return maxima

    def calculate_histogram(self):
        if self.image is None:
            return

        # Apply contrast and saturation adjustments
        adjusted_image = self.image.copy()
        
        # Apply contrast
        if self.contrast != 1.0:
            adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=self.contrast, beta=0)
        
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
        
        # Apply saturation
        if self.saturation != 50:
            sat_scale = self.saturation / 50.0
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], sat_scale)
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        
        adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.display_image(self.original_label, adjusted_image)

        total_pixels = hsv.shape[0] * hsv.shape[1]
        self.hue_areas = np.zeros(180)
        for h in range(180):
            mask = cv2.inRange(hsv, (h, 50, 50), (h, 255, 255))
            self.hue_areas[h] = np.sum(mask == 255) / total_pixels * 100

        smoothed_data = np.convolve(self.hue_areas, np.ones(2)/2, mode='same')

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.hue_areas, color='blue', alpha=0.3, label='Raw Distribution')
        ax.plot(smoothed_data, color='blue', linewidth=2, label='Smoothed Distribution')
        focus_range = (100, 179)

        # Mark ALL local maxima for visual reference (use RAW data for sensitivity)
        all_maxima = self.find_all_local_maxima(self.hue_areas, window_size=1)
        for idx in all_maxima:
            prominence = self.calculate_peak_prominence(self.hue_areas, idx)
            ax.plot(idx, self.hue_areas[idx], marker='^', color='orange', markersize=7, 
                   label='Local Maxima' if idx == all_maxima[0] else "")
            # Add prominence value as text
            ax.text(idx, self.hue_areas[idx] + 0.5, f'{prominence:.1f}', 
                   ha='center', va='bottom', fontsize=8, color='orange')

        # Find significant peaks in smoothed data with focus range
        peaks = self.find_significant_peaks(smoothed_data, threshold=0.2, min_distance=2, focus_range=focus_range)
        self.hue_segments = []
        for peak in peaks:
            left_valley = self.find_valleys(smoothed_data, peak, 'left', focus_range)
            right_valley = self.find_valleys(smoothed_data, peak, 'right', focus_range)
            min_segment_size = 1 if focus_range[0] <= peak <= focus_range[1] else 2
            if right_valley - left_valley >= min_segment_size:
                self.hue_segments.append((left_valley, right_valley))
        self.hue_segments.sort(key=lambda x: max(smoothed_data[x[0]:x[1]]), reverse=True)

        # Draw vertical lines at selected peaks
        for peak in peaks:
            prominence = self.calculate_peak_prominence(smoothed_data, peak)
            ax.axvline(peak, color='red', linestyle='--', alpha=0.8, 
                      label='Peak' if peak == peaks[0] else "")
            # Add prominence value as text
            ax.text(peak, smoothed_data[peak] + 0.5, f'P:{prominence:.1f}', 
                   ha='center', va='bottom', fontsize=8, color='red')

        # Highlight and label each segment
        for i, (start, end) in enumerate(self.hue_segments):
            hue_color = np.uint8([[[(start + end) // 2, 255, 255]]])
            rgb_color = cv2.cvtColor(hue_color, cv2.COLOR_HSV2RGB)[0][0] / 255.0
            ax.axvspan(start, end, color=rgb_color, alpha=0.3)
            segment_max = max(smoothed_data[start:end])
            label_x = (start + end) // 2
            label_y = segment_max + 0.5
            ax.text(label_x, label_y, f'Layer {i+1}\n({start}-{end})', 
                   ha='center', va='bottom', fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                   color='black')
            peak_idx = start + np.argmax(smoothed_data[start:end])
            ax.plot(peak_idx, smoothed_data[peak_idx], 'ro', markersize=4)

        ax.axvspan(focus_range[0], focus_range[1], color='yellow', alpha=0.1, label='Focus Range')
        ax.set_title("Hue Distribution with Color Layers\nColored areas show extracted layers based on peak prominence")
        ax.set_xlabel("Hue Value")
        ax.set_ylabel("Area Coverage (%)")
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right')
        self.canvas.draw()

    def extract_layers(self):
        if self.image is None or not self.hue_segments:
            return

        # Apply contrast and saturation adjustments
        adjusted_image = self.image.copy()
        
        # Apply contrast
        if self.contrast != 1.0:
            adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=self.contrast, beta=0)
        
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
        
        # Apply saturation
        if self.saturation != 50:
            # Convert saturation from 0-100 to 0-255 scale
            sat_scale = self.saturation / 50.0
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], sat_scale)
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)

        for i in reversed(range(self.segment_grid.count())):
            widget = self.segment_grid.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        for idx, (low, high) in enumerate(self.hue_segments):
            # Create container for each segment
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(5)

            # Create label for the segment
            label = QLabel(f"Layer {idx+1}\nHue {low}-{high}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid #ddd;")
            label.setFixedSize(200, 200)
            label.setCursor(Qt.PointingHandCursor)

            # Create the mask and extracted image
            lower_bound = np.array([low, 50, 50], dtype=np.uint8)
            upper_bound = np.array([high, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            extracted = np.where(mask[..., np.newaxis] == 255, adjusted_image, 255)
            pixmap = self.image_to_pixmap(extracted).scaled(200, 200, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)

            # Add click event for zoom
            label.mousePressEvent = lambda event, img=extracted: self.show_zoom_window(img)

            # Add to container
            container_layout.addWidget(label)
            self.segment_grid.addWidget(container, idx // 4, idx % 4)

    def show_zoom_window(self, image):
        # Convert numpy array to QPixmap
        pixmap = self.image_to_pixmap(image)
        # Create and show zoom window
        zoom_window = ZoomWindow(pixmap, self)
        zoom_window.exec_()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.image = cv2.imread(path)
            if self.image is not None:
                self.display_image(self.original_label, self.image)
                self.calculate_histogram()
                self.extract_btn.setEnabled(True)  # Enable extract button when image is loaded

    def calculate_peak_prominence(self, data, peak_idx, window_size=5):
        """Calculate how much a peak stands out relative to its surroundings"""
        # Get the window around the peak
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(data), peak_idx + window_size + 1)
        
        # Find the minimum value in the window
        window_min = min(data[start_idx:end_idx])
        
        # Calculate prominence as the difference between peak and minimum
        prominence = data[peak_idx] - window_min
        
        return prominence

    def find_significant_peaks(self, data, threshold=0.2, min_distance=2, focus_range=(100, 179)):
        """Find significant peaks in the histogram based on prominence and minimum distance"""
        peaks = []
        window_size = 3  # Smaller window for more local maxima
        
        for i in range(window_size, len(data)-window_size):
            # Check if it's a local maximum in the window
            if all(data[i] > data[i-j] for j in range(1, window_size+1)) and \
               all(data[i] > data[i+j] for j in range(1, window_size+1)):
                
                # Calculate peak prominence
                prominence = self.calculate_peak_prominence(data, i)
                
                # Apply different thresholds based on the focus range
                if focus_range[0] <= i <= focus_range[1]:
                    # Lower threshold for focus range
                    if prominence > threshold * 0.5:  # Adjusted threshold for prominence
                        peaks.append((i, prominence))
                else:
                    # Higher threshold for non-focus range
                    if prominence > threshold:
                        peaks.append((i, prominence))
        
        # Sort peaks by prominence
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Second pass: merge very close peaks
        merged_peaks = []
        if peaks:
            current_peak, current_prominence = peaks[0]
            
            for peak, prominence in peaks[1:]:
                # Use smaller min_distance for focus range
                current_min_distance = min_distance if focus_range[0] <= peak <= focus_range[1] else min_distance * 2
                
                if peak - current_peak < current_min_distance:
                    # If new peak has higher prominence, update current peak
                    if prominence > current_prominence:
                        current_peak = peak
                        current_prominence = prominence
                else:
                    # Add current peak and start new group
                    merged_peaks.append(current_peak)
                    current_peak = peak
                    current_prominence = prominence
            
            # Add the last peak
            merged_peaks.append(current_peak)
        
        return merged_peaks

    def find_valleys(self, data, peak, direction='left', focus_range=(100, 179)):
        """Find the valley (minimum) in a given direction from a peak"""
        if direction == 'left':
            valley = peak
            while valley > 0:
                if data[valley] <= data[valley-1]:
                    break
                valley -= 1
            # Look for a better valley in a small window
            # Use smaller window for focus range
            window_size = 2 if focus_range[0] <= peak <= focus_range[1] else 3
            search_start = max(0, valley - window_size)
            search_end = valley
            if search_end > search_start:
                valley = search_start + np.argmin(data[search_start:search_end])
        else:  # right
            valley = peak
            while valley < len(data)-1:
                if data[valley] <= data[valley+1]:
                    break
                valley += 1
            # Look for a better valley in a small window
            window_size = 2 if focus_range[0] <= peak <= focus_range[1] else 3
            search_start = valley
            search_end = min(len(data), valley + window_size)
            if search_end > search_start:
                valley = search_start + np.argmin(data[search_start:search_end])
        return valley

    def display_image(self, label, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def image_to_pixmap(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def extract_layers_all_maxima(self):
        if self.image is None:
            return

        # Apply contrast and saturation adjustments
        adjusted_image = self.image.copy()
        if self.contrast != 1.0:
            adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=self.contrast, beta=0)
        hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
        if self.saturation != 50:
            sat_scale = self.saturation / 50.0
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], sat_scale)
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)

        # Use the raw hue distribution (not smoothed)
        total_pixels = hsv.shape[0] * hsv.shape[1]
        hue_areas = np.zeros(180)
        for h in range(180):
            mask = cv2.inRange(hsv, (h, 50, 50), (h, 255, 255))
            hue_areas[h] = np.sum(mask == 255) / total_pixels * 100

        # Find all local maxima on the raw curve
        all_maxima = self.find_all_local_maxima(hue_areas, window_size=1)
        focus_range = (100, 179)

        hue_segments_all_maxima = []
        for peak in all_maxima:
            left_valley = self.find_valleys(hue_areas, peak, 'left', focus_range)
            right_valley = self.find_valleys(hue_areas, peak, 'right', focus_range)
            # For maxima in 100-179, always create a layer
            if 100 <= peak <= 179:
                if right_valley - left_valley < 1:
                    left_valley = peak
                    right_valley = min(179, peak+1)
                hue_segments_all_maxima.append((left_valley, right_valley))
            else:
                if right_valley - left_valley >= 1:
                    hue_segments_all_maxima.append((left_valley, right_valley))

        # Clear previous widgets
        for i in reversed(range(self.segment_grid.count())):
            widget = self.segment_grid.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Display all extracted layers
        for idx, (low, high) in enumerate(hue_segments_all_maxima):
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(5)
            label = QLabel(f"AllMax Layer {idx+1}\nHue {low}-{high}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid #ddd;")
            label.setFixedSize(200, 200)
            label.setCursor(Qt.PointingHandCursor)
            lower_bound = np.array([low, 50, 50], dtype=np.uint8)
            upper_bound = np.array([high, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            extracted = np.where(mask[..., np.newaxis] == 255, adjusted_image, 255)
            pixmap = self.image_to_pixmap(extracted).scaled(200, 200, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            label.mousePressEvent = lambda event, img=extracted: self.show_zoom_window(img)
            container_layout.addWidget(label)
            self.segment_grid.addWidget(container, idx // 4, idx % 4)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorExtractor()
    window.show()
    sys.exit(app.exec_())