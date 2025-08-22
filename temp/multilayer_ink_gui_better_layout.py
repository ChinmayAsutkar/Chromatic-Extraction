import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QLabel, QHBoxLayout, QScrollArea, QSizePolicy, QSplitter,
    QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def estimate_background_grid(image, grid_size=(50, 50)):
    h, w, _ = image.shape
    bg = np.zeros_like(image)
    gh, gw = grid_size
    for y in range(0, h, gh):
        for x in range(0, w, gw):
            patch = image[y:y+gh, x:x+gw]
            if patch.size > 0:
                med = np.median(patch.reshape(-1, 3), axis=0)
                bg[y:y+gh, x:x+gw] = np.full(patch.shape, med, dtype=np.uint8)
    return bg

def extract_hue_peaks(hsv_img, mask=None, smooth_sigma=3):
    hue = hsv_img[..., 0]
    if mask is not None:
        hue = hue[mask > 0]
    hist, _ = np.histogram(hue, bins=180, range=(0, 180))
    smoothed = gaussian_filter1d(hist, sigma=smooth_sigma)
    peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.05, distance=10)
    return peaks, hist

def generate_ink_layers(image, background, hue_peaks, band=10, sat_thresh=40, val_thresh=40):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    layers = []
    for peak in hue_peaks:
        lower = np.array([max(0, peak - band), sat_thresh, val_thresh])
        upper = np.array([min(179, peak + band), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        result = np.where(mask[..., None] == 255, image, background)
        layers.append((peak, result, mask))
    return layers

class MultiLayerInkGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Layer Handwritten Ink Extractor")
        self.setGeometry(50, 50, 1800, 1000)
        self.initUI()

    def initUI(self):
        w = QWidget()
        self.setCentralWidget(w)
        main_layout = QVBoxLayout(w)

        # Top button
        btn = QPushButton("Load Challan Image")
        btn.clicked.connect(self.load_image)
        main_layout.addWidget(btn)

        # Create grid layout for images
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)  # Add spacing between images

        # Create labels for images
        self.image_labels = []
        for i in range(8):  # 8 images total
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(400, 400)  # Set minimum size for each image
            label.setStyleSheet("border: 1px solid #cccccc;")  # Add border
            self.image_labels.append(label)
            row = i // 4  # 2 rows
            col = i % 4   # 4 columns
            self.grid_layout.addWidget(label, row, col)

        # Add grid widget to main layout
        main_layout.addWidget(self.grid_widget)

        # Add histogram at the bottom
        self.figure = plt.figure(figsize=(12, 3))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            return

        background = estimate_background_grid(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        strong_mask = ((sat > 50) & (val > 50)).astype(np.uint8) * 255

        hue_peaks, hist = extract_hue_peaks(hsv, mask=strong_mask)
        layers = generate_ink_layers(image, background, hue_peaks, band=10)

        # Display original image
        self.display_image(self.image_labels[0], image, "Original Image")

        # Display background
        self.display_image(self.image_labels[1], background, "Estimated Background")

        # Display up to 6 ink layers
        for i, (peak, layer_img, _) in enumerate(layers[:6], start=2):
            if i < 8:  # Ensure we don't exceed 8 images
                self.display_image(self.image_labels[i], layer_img, f"Ink Layer Hue {peak}")

        # Clear remaining labels if any
        for i in range(len(layers) + 2, 8):
            self.image_labels[i].clear()
            self.image_labels[i].setText("No Layer")

        # Update histogram
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(hist, color='purple')
        ax.set_title("Hue Histogram (Smoothed)")
        ax.set_xlabel("Hue Value")
        ax.set_ylabel("Frequency")
        for p in hue_peaks:
            ax.axvline(p, color='red', linestyle='--')
        self.canvas.draw()

    def display_image(self, label, img, title=""):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        
        # Calculate aspect ratio
        aspect_ratio = h / w
        target_width = 400  # Fixed width for grid
        target_height = int(target_width * aspect_ratio)
        
        # Scale image maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimg).scaled(
            target_width, 
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)
        label.setToolTip(title)  # Show title on hover

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MultiLayerInkGUI()
    win.show()
    sys.exit(app.exec_())
