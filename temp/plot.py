import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

# Blue ink HSV range
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])

def split_into_grids(image, grid_size=100):
    h, w = image.shape[:2]
    tile_h, tile_w = h // grid_size, w // grid_size
    grids = []
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * tile_h, (i + 1) * tile_h
            x1, x2 = j * tile_w, (j + 1) * tile_w
            grid = image[y1:y2, x1:x2]
            if grid.shape[0] > 0 and grid.shape[1] > 0:
                grids.append(grid)
            else:
                grids.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    return grids

def extract_blue_ink_stats(image, grid_size=100):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    blue_only = cv2.bitwise_and(hsv, hsv, mask=mask)
    grids = split_into_grids(blue_only, grid_size)
    hue_min = np.zeros((grid_size, grid_size), dtype=np.uint8)
    hue_max = np.zeros((grid_size, grid_size), dtype=np.uint8)
    sat_min = np.zeros((grid_size, grid_size), dtype=np.uint8)
    sat_max = np.zeros((grid_size, grid_size), dtype=np.uint8)
    all_blue_hues = []
    for idx, grid in enumerate(grids):
        i, j = divmod(idx, grid_size)
        h_vals = grid[:, :, 0].flatten()
        s_vals = grid[:, :, 1].flatten()
        h_vals = h_vals[h_vals > 0]
        s_vals = s_vals[s_vals > 0]
        if len(h_vals) == 0 or len(s_vals) == 0:
            hue_min[i, j] = 0
            hue_max[i, j] = 0
            sat_min[i, j] = 0
            sat_max[i, j] = 0
        else:
            hue_min[i, j] = np.min(h_vals)
            hue_max[i, j] = np.max(h_vals)
            sat_min[i, j] = np.min(s_vals)
            sat_max[i, j] = np.max(s_vals)
            all_blue_hues.extend(h_vals)
    return hue_min, hue_max, sat_min, sat_max, np.array(all_blue_hues)

def get_nonzero_blue_regions(hue_min, hue_max, sat_min, sat_max):
    blue_regions = []
    grid_size = hue_min.shape[0]
    for i in range(grid_size):
        for j in range(grid_size):
            if hue_min[i, j] > 0 or hue_max[i, j] > 0 or sat_min[i, j] > 0 or sat_max[i, j] > 0:
                blue_regions.append({
                    'grid': (i, j),
                    'hue_min': int(hue_min[i, j]),
                    'hue_max': int(hue_max[i, j]),
                    'sat_min': int(sat_min[i, j]),
                    'sat_max': int(sat_max[i, j])
                })
    return blue_regions

def fit_curve(x, y, fit_type):
    mask = y > 0
    x_filtered = x[mask]
    y_filtered = y[mask]
    if len(x_filtered) < 3:
        return x, y, []
    try:
        if fit_type == 'Polynomial':
            coeffs = np.polyfit(x_filtered, y_filtered, min(8, len(x_filtered)-1))
            y_fit = np.polyval(coeffs, x)
        elif fit_type == 'Gaussian':
            def gaussian(x, a, b, c):
                return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))
            popt, _ = curve_fit(gaussian, x_filtered, y_filtered,
                                p0=[y_filtered.max(), x_filtered[np.argmax(y_filtered)], 10])
            y_fit = gaussian(x, *popt)
        elif fit_type == 'Spline':
            spline = UnivariateSpline(x_filtered, y_filtered, k=3, s=0)
            y_fit = spline(x)
        elif fit_type == 'Exponential':
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            popt, _ = curve_fit(exp_func, x_filtered, y_filtered,
                                p0=[y_filtered.max(), 0.01, 0])
            y_fit = exp_func(x, *popt)
        elif fit_type == 'Power':
            def power_func(x, a, b):
                return a * np.power(np.maximum(x, 0.1), b)
            popt, _ = curve_fit(power_func, x_filtered, y_filtered,
                                p0=[y_filtered.max(), -1])
            y_fit = power_func(x, *popt)
        else:
            y_fit = y
        peaks, _ = find_peaks(y_fit, height=np.max(y_fit)*0.1, distance=3)
        valleys, _ = find_peaks(-y_fit, height=-np.max(y_fit)*0.9, distance=3)
        return x, y_fit, {'peaks': peaks, 'valleys': valleys}
    except Exception as e:
        print(f"Curve fitting error for {fit_type}: {e}")
        return x, y, []

def plot_surface(hue_min, hue_max, sat_min, sat_max, title_prefix=''):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(hue_min, cmap='hsv', vmin=0, vmax=180)
    plt.colorbar(label='Hue (min)')
    plt.title(f'{title_prefix} Hue Min Surface')
    plt.subplot(2, 2, 2)
    plt.imshow(hue_max, cmap='hsv', vmin=0, vmax=180)
    plt.colorbar(label='Hue (max)')
    plt.title(f'{title_prefix} Hue Max Surface')
    plt.subplot(2, 2, 3)
    plt.imshow(sat_min, cmap='gray', vmin=0, vmax=255)
    plt.colorbar(label='Saturation (min)')
    plt.title(f'{title_prefix} Saturation Min Surface')
    plt.subplot(2, 2, 4)
    plt.imshow(sat_max, cmap='gray', vmin=0, vmax=255)
    plt.colorbar(label='Saturation (max)')
    plt.title(f'{title_prefix} Saturation Max Surface')
    plt.tight_layout()
    plt.show()

class BlueInkAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title('Blue Ink Analysis with Advanced Curve Fitting')
        self.root.geometry('1400x900')
        self.image_path = None
        self.blue_hues = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        upload_frame = ttk.LabelFrame(control_frame, text="Image Upload", padding=10)
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        self.upload_button = ttk.Button(upload_frame, text='Upload Image',
                                        command=self.upload_image)
        self.upload_button.pack(pady=5)
        self.image_label = tk.Label(upload_frame, text="No image selected",
                                    width=30, height=15, bg='lightgray')
        self.image_label.pack(pady=5)
        fitting_frame = ttk.LabelFrame(control_frame, text="Curve Fitting Options", padding=10)
        fitting_frame.pack(fill=tk.X, pady=(0, 10))
        fit_types = ['Polynomial', 'Gaussian', 'Spline', 'Exponential', 'Power']
        self.fit_buttons = {}
        for fit_type in fit_types:
            btn = ttk.Button(fitting_frame, text=f'{fit_type} Fit',
                             command=lambda ft=fit_type: self.plot_line_histogram(ft))
            btn.pack(fill=tk.X, pady=2)
            self.fit_buttons[fit_type] = btn
            btn.configure(state='disabled')
        process_frame = ttk.LabelFrame(control_frame, text="Analysis", padding=10)
        process_frame.pack(fill=tk.X, pady=(0, 10))
        self.process_button = ttk.Button(process_frame, text='Analyze Blue Ink',
                                         command=self.process_image, state='disabled')
        self.process_button.pack(pady=5)
        self.surface_button = ttk.Button(process_frame, text='Show Surface Plots',
                                         command=self.show_surface_plots, state='disabled')
        self.surface_button.pack(pady=5)
        info_frame = ttk.LabelFrame(control_frame, text="Analysis Results", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)
        self.info_text = tk.Text(info_frame, height=10, width=40)
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp *.tiff')]
        )
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((250, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk
            self.process_button.configure(state='normal')

    def process_image(self):
        if not self.image_path:
            return
        image = cv2.imread(self.image_path)
        self.hue_min, self.hue_max, self.sat_min, self.sat_max, self.blue_hues = extract_blue_ink_stats(image)
        for btn in self.fit_buttons.values():
            btn.configure(state='normal')
        self.surface_button.configure(state='normal')
        self.update_info()
        self.plot_line_histogram('Raw')

    def update_info(self):
        if self.blue_hues is None or len(self.blue_hues) == 0:
            return
        blue_regions = get_nonzero_blue_regions(self.hue_min, self.hue_max, self.sat_min, self.sat_max)
        info_text = f"""Blue Ink Analysis Results:
Total blue pixels: {len(self.blue_hues)}
Hue range: {np.min(self.blue_hues):.1f} - {np.max(self.blue_hues):.1f}
Mean hue: {np.mean(self.blue_hues):.1f}
Std deviation: {np.std(self.blue_hues):.1f}

Grid Analysis:
Non-zero blue regions (grid, hue_min, hue_max, sat_min, sat_max):
"""
        for region in blue_regions:
            info_text += f"Grid {region['grid']}: Hmin={region['hue_min']}, Hmax={region['hue_max']}, Smin={region['sat_min']}, Smax={region['sat_max']}\n"
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)

    def plot_line_histogram(self, fit_type):
        if self.blue_hues is None or len(self.blue_hues) == 0:
            return
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(10, 6))
        hist_counts, bin_edges = np.histogram(self.blue_hues, bins=60, range=(90, 130))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, hist_counts, 'o-', linewidth=1, markersize=3,
                color='blue', alpha=0.7, label='Blue Ink Distribution')
        if fit_type != 'Raw':
            x_fit, y_fit, extrema = fit_curve(bin_centers, hist_counts, fit_type)
            ax.plot(x_fit, y_fit, '-', linewidth=2, color='red', alpha=0.8,
                    label=f'{fit_type} Fit')
            if extrema and 'peaks' in extrema:
                peaks = extrema['peaks']
                if len(peaks) > 0:
                    ax.plot(x_fit[peaks], y_fit[peaks], '^', markersize=8,
                            color='green', label=f'Maxima ({len(peaks)})')
                    for peak in peaks:
                        ax.annotate(f'Max: {x_fit[peak]:.1f}',
                                    xy=(x_fit[peak], y_fit[peak]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, ha='left')
            if extrema and 'valleys' in extrema:
                valleys = extrema['valleys']
                if len(valleys) > 0:
                    ax.plot(x_fit[valleys], y_fit[valleys], 'v', markersize=8,
                            color='orange', label=f'Minima ({len(valleys)})')
                    for valley in valleys:
                        ax.annotate(f'Min: {x_fit[valley]:.1f}',
                                    xy=(x_fit[valley], y_fit[valley]),
                                    xytext=(5, -15), textcoords='offset points',
                                    fontsize=8, ha='left')
        ax.set_xlabel('Hue Value')
        ax.set_ylabel('Pixel Count')
        ax.set_title(f'Blue Ink Hue Distribution - {fit_type} Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_surface_plots(self):
        if hasattr(self, 'hue_min'):
            plot_surface(self.hue_min, self.hue_max, self.sat_min, self.sat_max, title_prefix='Blue Ink')

if __name__ == '__main__':
    root = tk.Tk()
    app = BlueInkAnalyzer(root)
    root.mainloop()

