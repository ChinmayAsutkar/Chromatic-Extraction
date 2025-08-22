import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QSpinBox, QHBoxLayout, QInputDialog, QScrollArea, QGridLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
import mplcursors
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

class HuePeakAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hue Peak Histogram Analyzer")
        self.setGeometry(200, 200, 1600, 1000)
        self.image = None
        self.original_bgr = None
        self.grid_size = 20
        self.h_list, self.s_list, self.v_list = [], [], []
        self.layout = QVBoxLayout(self)

        control_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_button)

        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(5, 100)
        self.grid_spin.setValue(20)
        self.grid_spin.setPrefix("Grid px: ")
        self.grid_spin.valueChanged.connect(self.update_grid_size)
        control_layout.addWidget(self.grid_spin)

        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(1, 30)
        self.bin_spin.setValue(4)
        self.bin_spin.setPrefix("Hue bin: ")
        control_layout.addWidget(self.bin_spin)

        self.analyze_button = QPushButton("Analyze Hue Histogram")
        self.analyze_button.clicked.connect(self.analyze_hue_histogram)
        control_layout.addWidget(self.analyze_button)

        self.sv_button = QPushButton("Analyze S/V for Hue Range")
        self.sv_button.clicked.connect(self.analyze_sv_for_range)
        self.sv_button.setEnabled(False)
        control_layout.addWidget(self.sv_button)

        self.layout.addLayout(control_layout)

        image_grid = QGridLayout()
        self.image_label = QLabel("Loaded image will appear here")
        self.image_label.setFixedSize(400, 300)
        image_grid.addWidget(self.image_label, 0, 0)

        self.bg_label = QLabel("Background")
        self.bg_label.setFixedSize(400, 300)
        image_grid.addWidget(self.bg_label, 0, 1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.ink_container = QWidget()
        self.ink_layout = QHBoxLayout(self.ink_container)
        self.scroll_area.setWidget(self.ink_container)
        image_grid.addWidget(self.scroll_area, 1, 0, 1, 2)

        self.layout.addLayout(image_grid)

        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # QLabel for spline equations (use a standard font)
        self.equation_label = QLabel("")
        self.equation_label.setWordWrap(True)
        self.equation_label.setFont(QFont('Arial', 10))  # Use Arial to avoid missing font warning
        self.layout.addWidget(self.equation_label)

    def update_grid_size(self):
        self.grid_size = self.grid_spin.value()

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Image')
        if fname:
            bgr = cv2.imread(fname)
            self.original_bgr = bgr.copy()
            self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            self.setWindowTitle(f"Loaded: {fname.split('/')[-1]}")
            self.display_loaded_image()
            self.extract_and_display_layers()

    def display_loaded_image(self):
        self.set_pixmap_from_bgr(self.original_bgr, self.image_label)

    def set_pixmap_from_bgr(self, bgr_image, label_widget):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(label_widget.size(), aspectRatioMode=1)
        label_widget.setPixmap(pixmap)

    def estimate_background_grid(self, image, grid_size=(30, 30)):
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

    def extract_hue_peaks(self, hsv_img, mask=None, smooth_sigma=3):
        hue = hsv_img[..., 0]
        if mask is not None:
            hue = hue[mask > 0]
        hist, _ = np.histogram(hue, bins=180, range=(0, 180))
        smoothed = gaussian_filter1d(hist, sigma=smooth_sigma)
        peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.05, distance=10)
        return peaks, hist

    def extract_and_display_layers(self):
        if self.original_bgr is None:
            return
        image = self.original_bgr
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        background = self.estimate_background_grid(image)
        self.set_pixmap_from_bgr(background, self.bg_label)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        strong_mask = ((sat > 30) & (val > 30)).astype(np.uint8) * 255
        peaks, hist = self.extract_hue_peaks(hsv, mask=strong_mask)
        if 120 not in peaks:
            peaks = np.append(peaks, 120)
        band = 15
        self.clear_layout(self.ink_layout)
        for peak in peaks:
            lower = np.array([max(0, peak - band), 40, 40])
            upper = np.array([min(179, peak + band), 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            result = np.where(mask[..., None] == 255, image, background)
            lbl = QLabel(f"Ink Layer {peak}")
            lbl.setFixedSize(400, 300)
            self.set_pixmap_from_bgr(result, lbl)
            self.ink_layout.addWidget(lbl)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(hist, color='purple')
        ax.set_title("Hue Histogram (Smoothed)")
        ax.set_xlabel("Hue Value")
        ax.set_ylabel("Frequency (Pixel Count)")
        ax.set_yscale('log')
        for p in peaks:
            ax.axvline(p, color='red', linestyle='--')
        self.canvas.draw()

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def analyze_hue_histogram(self):
        if self.image is None:
            return
        hsv = self.image
        self.h_list, self.s_list, self.v_list = [], [], []
        h, w = hsv.shape[:2]
        for y in range(0, h, self.grid_size):
            for x in range(0, w, self.grid_size):
                patch = hsv[y:y+self.grid_size, x:x+self.grid_size]
                if patch.size > 0:
                    self.h_list.append(np.median(patch[..., 0]))
                    self.s_list.append(np.median(patch[..., 1]))
                    self.v_list.append(np.median(patch[..., 2]))
        h_array = np.array(self.h_list)
        bin_width = self.bin_spin.value()
        bin_edges = np.arange(0, 181, bin_width)
        counts, edges = np.histogram(h_array, bins=bin_edges)
        centers = (edges[:-1] + edges[1:]) / 2
        peaks, _ = find_peaks(counts, height=5, distance=2)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        bars = []
        for i in range(len(centers)):
            center = centers[i]
            height = counts[i]
            hsv_color = np.uint8([[[int(center), 255, 255]]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0] / 255
            bar = ax.bar(center, height, width=bin_width, color=rgb_color)
            bars.extend(bar)
        ax.plot(centers[peaks], counts[peaks], 'rx', label='Peaks')
        ax.set_title("Hue Histogram with Peaks")
        ax.set_xlabel("Hue")
        ax.set_ylabel("Grid Count")
        ax.set_yscale('log')
        ax.legend()
        cursor = mplcursors.cursor(bars, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(f"Hue: {sel.target[0]:.1f}\nCount: {int(sel.target[1])}")
        self.canvas.draw()
        self.sv_button.setEnabled(True)

    def analyze_sv_for_range(self):
        if not self.h_list:
            return
        h_array = np.array(self.h_list)
        s_array = np.array(self.s_list)
        v_array = np.array(self.v_list)
        hue_input, ok = QInputDialog.getText(self, "Hue Range", "Enter hue range (e.g. 160-180):")
        if not ok or '-' not in hue_input:
            return
        try:
            hmin, hmax = map(int, hue_input.split('-'))
        except:
            return
        mask = (h_array >= hmin) & (h_array <= hmax)
        if np.sum(mask) == 0:
            return
        s_vals = s_array[mask]
        v_vals = v_array[mask]

        # Use tight_layout instead of constrained_layout to avoid warning
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        fig.tight_layout(pad=4.0)

        # Saturation
        counts_s, bins_s = np.histogram(s_vals, bins='auto')
        bin_centers_s = (bins_s[:-1] + bins_s[1:]) / 2
        axs[0].hist(s_vals, bins=bins_s, color='orange', alpha=0.7, log=True)
        axs[0].set_title(f"Saturation Distribution for H:{hmin}-{hmax}")
        axs[0].set_xlabel("Saturation")
        axs[0].set_ylabel("Grid Count")

        # Value
        counts_v, bins_v = np.histogram(v_vals, bins='auto')
        bin_centers_v = (bins_v[:-1] + bins_v[1:]) / 2
        axs[1].hist(v_vals, bins=bins_v, color='blue', alpha=0.7, log=True)
        axs[1].set_title(f"Value Distribution for H:{hmin}-{hmax}")
        axs[1].set_xlabel("Value")
        axs[1].set_ylabel("Grid Count")

        # Add spline and full equation
        eqn_s = self.add_spline_equation(axs[0], bin_centers_s, counts_s, "Saturation")
        eqn_v = self.add_spline_equation(axs[1], bin_centers_v, counts_v, "Value")

        # Update the UI label with both equations
        eqn_text = ""
        if eqn_s:
            eqn_text += f"<b>Saturation Spline:</b><br>{eqn_s}<br><br>"
        if eqn_v:
            eqn_text += f"<b>Value Spline:</b><br>{eqn_v}"
        self.equation_label.setText(eqn_text)

        mplcursors.cursor(axs[0].containers[0], hover=True)
        mplcursors.cursor(axs[1].containers[0], hover=True)
        plt.show()

    def add_spline_equation(self, ax, bin_centers, counts, plot_name):
        threshold = 10
        valid_mask = counts >= threshold
        if np.sum(valid_mask) < 4:  # Need at least 4 points for cubic spline
            return ""
        x = bin_centers[valid_mask]
        y = counts[valid_mask]
        try:
            degree = 3  # Set spline degree explicitly
            spline = UnivariateSpline(x, y, k=degree, s=len(x)*1.5)
            x_smooth = np.linspace(min(x), max(x), 100)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, 'r-', lw=2, label='Spline Fit')
            knots = spline.get_knots()
            coeffs = spline.get_coeffs()
            # Build the explicit equation string
            terms = [f"{c:.2f}·B<sub>{i},{degree}</sub>(x)" for i, c in enumerate(coeffs)]
            eqn = "S(x) = " + " + ".join(terms)
            knots_str = ", ".join([f"{k:.2f}" for k in knots])
            coeffs_str = ", ".join([f"{c:.2f}" for c in coeffs])
            equation = (f"{eqn}<br>"
                        f"<b>Knots:</b> [{knots_str}]<br>"
                        f"<b>Coefficients:</b> [{coeffs_str}]")
            # Display equation below the plot
            ax.text(0.5, -0.35, equation.replace("<br>", "\n"), transform=ax.transAxes,
                    fontsize=9, ha='center', va='top',
                    bbox=dict(facecolor='white', alpha=0.8))
            return equation
        except Exception as e:
            print(f"Spline error: {e}")
            return ""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont('Arial', 10))  # Set default font to avoid missing font warning
    viewer = HuePeakAnalyzer()
    viewer.show()
    sys.exit(app.exec_())

