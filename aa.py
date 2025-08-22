import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QScrollArea, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

class AutoInkExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Ink Layer Extractor")
        self.setGeometry(100, 100, 1200, 800)
        self.image = None

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        self.canvas = FigureCanvas(Figure(figsize=(6, 3)))
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

        self.scroll_area = QScrollArea()
        self.scroll_content = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.image = cv2.imread(path)
            self.extract_layers()

    def extract_layers(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mask = (v >= 50) & (s >= 30)
        filtered_hue = h[mask]

        raw_hist, _ = np.histogram(filtered_hue, bins=180, range=(0, 180))
        hist_smoothed = gaussian_filter1d(raw_hist, sigma=2)
        peaks, _ = find_peaks(hist_smoothed, height=np.max(hist_smoothed) * 0.05)

        log_hist = np.log1p(hist_smoothed)

        self.ax.clear()
        self.ax.plot(log_hist, label='Log(Smoothed Hue Histogram)')

        for peak in peaks:
            h_left = self.expand_peak(log_hist, peak, direction="left")
            h_right = self.expand_peak(log_hist, peak, direction="right")
            self.ax.axvline(x=h_left, color='green', linestyle='--')
            self.ax.axvline(x=h_right, color='green', linestyle='--')
            self.ax.axvline(x=peak, color='red', linestyle='-')

            hue_mask = (h >= h_left) & (h <= h_right)
            combined_mask = hue_mask & mask
            final_mask = combined_mask.astype(np.uint8) * 255

            white_bg = np.ones_like(self.image, dtype=np.uint8) * 255
            result = np.where(final_mask[..., None] == 255, self.image, white_bg)

            pixmap = self.cv2_to_pixmap(result).scaled(300, 300, Qt.KeepAspectRatio)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setToolTip(f"Hue ~{peak} [{h_left}, {h_right}]")
            self.scroll_layout.addWidget(label)

        self.ax.set_title("Log(Area) vs Hue")
        self.ax.set_xlabel("Hue")
        self.ax.set_ylabel("log(Frequency + 1)")
        self.ax.legend()
        self.canvas.draw()

    def expand_peak(hist, peak_idx, direction="right", max_width=20, threshold_ratio=0.3, min_width=8):
        """
        Expands the hue range from a histogram peak until area growth becomes significant.
        Also ensures minimum range width.
        """
        if direction == "right":
            idx_seq = range(peak_idx + 1, min(peak_idx + max_width + 1, len(hist)))
        else:
            idx_seq = range(peak_idx - 1, max(peak_idx - max_width - 1, -1), -1)

        area = hist[peak_idx]
        selected_indices = [peak_idx]

        for i in idx_seq:
            new_area = area + hist[i]
            growth_ratio = (new_area - area) / (area + 1e-6)

            # Only allow breaking after min_width is met
            if len(selected_indices) >= min_width and growth_ratio > threshold_ratio:
                break

            if direction == "right":
                selected_indices.append(i)
            else:
                selected_indices.insert(0, i)

            area = new_area

        return selected_indices[0] if direction == "left" else selected_indices[-1]

    def cv2_to_pixmap(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QPixmap(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AutoInkExtractor()
    window.show()
    sys.exit(app.exec_())