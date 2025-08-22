import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QFileDialog, QSlider, QHBoxLayout, QGroupBox, QGridLayout, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def cv2_to_pixmap(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QPixmap(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))


class HistogramPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_histogram(self, raw_hist, smoothed_hist, peaks=None):
        self.ax.clear()
        self.ax.plot(raw_hist, label="Raw Hue Histogram", color='gray', linestyle='--')
        self.ax.plot(smoothed_hist, label="Smoothed Histogram", color='blue')
        if peaks is not None:
            self.ax.plot(peaks, smoothed_hist[peaks], 'rx', label='Detected Peaks')
        self.ax.set_title("Hue Histogram in V Range")
        self.ax.set_xlabel("Hue (0–179)")
        self.ax.set_ylabel("Frequency")
        self.ax.legend()
        self.draw()


class InkAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ink Layer Analyzer (Hue by Value)")
        self.image = None
        self.hsv = None
        self.v_min = 80
        self.v_max = 160

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Buttons and labels
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        self.status_label = QLabel("No image loaded.")
        self.vmin_label = QLabel(f"V min: {self.v_min}")
        self.vmax_label = QLabel(f"V max: {self.v_max}")

        # V sliders
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, 255)
        self.vmin_slider.setValue(self.v_min)
        self.vmin_slider.valueChanged.connect(self.update_vmin)

        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmax_slider.setRange(0, 255)
        self.vmax_slider.setValue(self.v_max)
        self.vmax_slider.valueChanged.connect(self.update_vmax)

        # Group sliders
        sliders_layout = QGridLayout()
        sliders_layout.addWidget(self.vmin_label, 0, 0)
        sliders_layout.addWidget(self.vmin_slider, 0, 1)
        sliders_layout.addWidget(self.vmax_label, 1, 0)
        sliders_layout.addWidget(self.vmax_slider, 1, 1)
        sliders_group = QGroupBox("Value Range")
        sliders_group.setLayout(sliders_layout)

        # Plot area
        self.plot_canvas = HistogramPlotCanvas(self)

        # Output layer previews
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(sliders_group)
        layout.addWidget(self.plot_canvas)
        layout.addWidget(QLabel("Extracted Ink Layers:"))
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)
        self.resize(900, 800)

    def update_vmin(self, value):
        self.v_min = value
        self.vmin_label.setText(f"V min: {value}")
        self.process_image()

    def update_vmax(self, value):
        self.v_max = value
        self.vmax_label.setText(f"V max: {value}")
        self.process_image()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            self.status_label.setText(f"Loaded: {file_name}")
            self.process_image()

    def process_image(self):
        if self.hsv is None:
            return

        h, s, v = cv2.split(self.hsv)
        v_mask = (v >= self.v_min) & (v <= self.v_max)
        filtered_hue = h[v_mask]

        if len(filtered_hue) == 0:
            self.plot_canvas.plot_histogram(np.zeros(180), np.zeros(180))
            return

        raw_hist, _ = np.histogram(filtered_hue, bins=180, range=(0, 180))
        smoothed_hist = gaussian_filter1d(raw_hist, sigma=2)
        peaks, _ = find_peaks(smoothed_hist, height=np.max(smoothed_hist) * 0.1)

        # Update histogram plot
        self.plot_canvas.plot_histogram(raw_hist, smoothed_hist, peaks)

        # Clear previous previews
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Extract and display layers
        for i, h0 in enumerate(peaks):
            hue_thresh = smoothed_hist[h0] * 0.25
            h_l = h0
            while h_l > 0 and smoothed_hist[h_l] >= hue_thresh and h0 - h_l <= 15:
                h_l -= 1
            h_r = h0
            while h_r < 179 and smoothed_hist[h_r] >= hue_thresh and h_r - h0 <= 15:
                h_r += 1

            final_mask = (
                (v >= self.v_min) & (v <= self.v_max) &
                (h >= h_l) & (h <= h_r)
            ).astype(np.uint8) * 255

            # Use white background instead of black
            white_bg = np.ones_like(self.image, dtype=np.uint8) * 255
            result = np.where(final_mask[..., None] == 255, self.image, white_bg)

            pixmap = cv2_to_pixmap(result)

            layer_label = QLabel(f"Layer {i+1} (Hue ~{h0}, range {h_l}–{h_r})")
            img_label = QLabel()
            img_label.setPixmap(pixmap.scaledToWidth(300))

            self.scroll_layout.addWidget(layer_label)
            self.scroll_layout.addWidget(img_label)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InkAnalyzer()
    window.show()
    sys.exit(app.exec_())