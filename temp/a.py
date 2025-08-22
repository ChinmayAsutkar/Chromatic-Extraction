import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import matplotlib.colors as mcolors

# Global HSV range - now will be updated by UI
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])

def split_into_grids(image, grid_size=100):
    h, w = image.shape[:2]
    # Calculate tile dimensions, ensuring no division by zero for very small images
    tile_h = h // grid_size if grid_size > 0 else h
    tile_w = w // grid_size if grid_size > 0 else w

    if tile_h == 0: tile_h = 1 # Ensure minimum tile size
    if tile_w == 0: tile_w = 1 # Ensure minimum tile size

    grids = []
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * tile_h, min((i + 1) * tile_h, h)
            x1, x2 = j * tile_w, min((j + 1) * tile_w, w)
            grid = image[y1:y2, x1:x2]
            if grid.shape[0] > 0 and grid.shape[1] > 0:
                grids.append(grid)
            else:
                # Append an empty grid of the expected size or handle as needed
                # For this application, a small zero array is fine, it will be filtered later
                grids.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8)) 
    return grids

def extract_blue_ink_stats(image, grid_size=100, lower_blue_hsv=LOWER_BLUE, upper_blue_hsv=UPPER_BLUE):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue_hsv, upper_blue_hsv)
    blue_only = cv2.bitwise_and(hsv, hsv, mask=mask)
    grids = split_into_grids(blue_only, grid_size)
    
    # Initialize arrays based on grid_size for consistency, even if dimensions are 1
    # Ensure grid_size is at least 1 to prevent shape errors
    actual_grid_size = max(1, grid_size)
    hue_min = np.zeros((actual_grid_size, actual_grid_size), dtype=np.uint8)
    hue_max = np.zeros((actual_grid_size, actual_grid_size), dtype=np.uint8)
    sat_min = np.zeros((actual_grid_size, actual_grid_size), dtype=np.uint8)
    sat_max = np.zeros((actual_grid_size, actual_grid_size), dtype=np.uint8)
    all_blue_hues = []
    
    for idx, grid in enumerate(grids):
        # Calculate i, j based on actual_grid_size to map to the correct location in the 2D arrays
        i, j = divmod(idx, actual_grid_size)

        if i >= actual_grid_size or j >= actual_grid_size: # safety check for grids that might exceed expected size
            continue

        h_vals = grid[:, :, 0].flatten()
        s_vals = grid[:, :, 1].flatten()
        
        # Filter out zero values (from masked regions or empty grids)
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
        return x, np.zeros_like(y), {'peaks': [], 'valleys': []} # Return empty extrema for too few points
    
    try:
        if fit_type == 'Polynomial':
            # Ensure degree does not exceed number of points - 1
            degree = min(8, len(x_filtered) - 1)
            if degree < 1: # Need at least degree 1 for polyfit
                return x, np.zeros_like(y), {'peaks': [], 'valleys': []}
            coeffs = np.polyfit(x_filtered, y_filtered, degree)
            y_fit = np.polyval(coeffs, x)
        elif fit_type == 'Gaussian':
            def gaussian(x, a, b, c):
                return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))
            # Provide better initial guess if possible
            p0 = [y_filtered.max(), x_filtered[np.argmax(y_filtered)], (x_filtered.max() - x_filtered.min()) / 4]
            if p0[2] == 0: p0[2] = 1 # Avoid division by zero for c
            popt, _ = curve_fit(gaussian, x_filtered, y_filtered, p0=p0)
            y_fit = gaussian(x, *popt)
        elif fit_type == 'Spline':
            spline = UnivariateSpline(x_filtered, y_filtered, k=min(3, len(x_filtered)-1), s=0)
            y_fit = spline(x)
        elif fit_type == 'Exponential':
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            # Adjust p0 for more robust fitting
            p0=[y_filtered.max(), 0.01, y_filtered.min()]
            popt, _ = curve_fit(exp_func, x_filtered, y_filtered, p0=p0, maxfev=5000)
            y_fit = exp_func(x, *popt)
        elif fit_type == 'Power':
            def power_func(x, a, b):
                # Ensure x is positive for power function
                return a * np.power(np.maximum(x, 1e-6), b) # Use 1e-6 instead of 0.1 for better stability
            popt, _ = curve_fit(power_func, x_filtered, y_filtered, p0=[y_filtered.max(), -1], maxfev=5000)
            y_fit = power_func(x, *popt)
        else:
            y_fit = y # Should not happen with controlled fit_type
            
        # Find peaks and valleys on the fitted curve, ensuring min height is reasonable
        min_height_peak = np.max(y_fit) * 0.1
        min_height_valley = -np.max(y_fit) * 0.9 if np.max(y_fit) > 0 else -1 # Ensure it's negative for valleys
        
        peaks, _ = find_peaks(y_fit, height=min_height_peak, distance=3)
        valleys, _ = find_peaks(-y_fit, height=min_height_valley, distance=3) # Use -y_fit for valleys
        
        return x, y_fit, {'peaks': peaks, 'valleys': valleys}
    except Exception as e:
        print(f"Curve fitting error for {fit_type}: {e}")
        return x, np.zeros_like(y), {'peaks': [], 'valleys': []} # Return empty extrema on error

def plot_surface(hue_min, hue_max, sat_min, sat_max, title_prefix=''):
    plt.figure(figsize=(12, 10))
    
    # Filter out 0 values for better visualization, as 0 can mean no blue pixels
    h_min_filtered = hue_min[hue_min > 0]
    h_max_filtered = hue_max[hue_max > 0]
    s_min_filtered = sat_min[sat_min > 0]
    s_max_filtered = sat_max[sat_max > 0]

    # Create 2D histograms for density visualization
    plt.subplot(2, 2, 1)
    plt.hist2d(h_min_filtered.flatten(), s_min_filtered.flatten(), bins=50, 
               range=[[0, 180], [0, 255]], cmap='hsv', norm=mcolors.LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Hue (min)')
    plt.ylabel('Saturation (min)')
    plt.title(f'{title_prefix} Hue Min Surface Density')

    plt.subplot(2, 2, 2)
    plt.hist2d(h_max_filtered.flatten(), s_max_filtered.flatten(), bins=50, 
               range=[[0, 180], [0, 255]], cmap='hsv', norm=mcolors.LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Hue (max)')
    plt.ylabel('Saturation (max)')
    plt.title(f'{title_prefix} Hue Max Surface Density')

    plt.subplot(2, 2, 3)
    plt.hist2d(h_min_filtered.flatten(), s_min_filtered.flatten(), bins=50, 
               range=[[0, 180], [0, 255]], cmap='gray', norm=mcolors.LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Hue (min)')
    plt.ylabel('Saturation (min)')
    plt.title(f'{title_prefix} Saturation Min Surface Density')

    plt.subplot(2, 2, 4)
    plt.hist2d(h_max_filtered.flatten(), s_max_filtered.flatten(), bins=50, 
               range=[[0, 180], [0, 255]], cmap='gray', norm=mcolors.LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Hue (max)')
    plt.ylabel('Saturation (max)')
    plt.title(f'{title_prefix} Saturation Max Surface Density')

    plt.tight_layout()
    plt.show()

# --- NEW FEATURE: Enhanced 3D Plotting with Controls ---
def create_3d_plot_window(parent, hue_data_raw, sat_data_raw, plot_type):
    window = tk.Toplevel(parent)
    window.title(f"3D {plot_type} Hue-Saturation Density")
    window.geometry("800x600")
    
    fig = plt.figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Filter out 0 values from the data to prevent them from skewing the histogram
    valid = (hue_data_raw.flatten() > 0) & (sat_data_raw.flatten() > 0)
    hue_data = hue_data_raw.flatten()[valid]
    sat_data = sat_data_raw.flatten()[valid]

    if not hue_data.size or not sat_data.size:
        ax.text2D(0.5, 0.5, "No valid blue pixels found for this plot.", transform=ax.transAxes, ha='center')
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return

    # Plot density surface
    hist, xedges, yedges = np.histogram2d(
        hue_data, 
        sat_data,
        bins=50, range=[[90, 130], [0, 255]] # Use fixed range for consistency across plots
    )
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    surf = ax.plot_surface(X, Y, hist.T, cmap='viridis' if 'Min' in plot_type else 'plasma',
                          rstride=1, cstride=1, alpha=0.8)
    
    ax.set_xlabel('Hue Value')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Grid Count')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Embed in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    
    # Toolbar
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    
    # Control Frame
    control_frame = ttk.Frame(window)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # View Controls
    ttk.Label(control_frame, text="View Controls:").pack(side=tk.LEFT)
    
    def update_view(elev=None, azim=None, dist=None):
        if elev is not None: ax.elev = elev
        if azim is not None: ax.azim = azim
        if dist is not None: ax.dist = dist
        canvas.draw()
    
    # Elevation Control
    ttk.Scale(control_frame, from_=0, to=90, orient=tk.HORIZONTAL,
             command=lambda v: update_view(elev=float(v))).pack(side=tk.LEFT, padx=5)
    ttk.Label(control_frame, text="Elev").pack(side=tk.LEFT)
    
    # Azimuth Control
    ttk.Scale(control_frame, from_=0, to=360, orient=tk.HORIZONTAL,
             command=lambda v: update_view(azim=float(v))).pack(side=tk.LEFT, padx=5)
    ttk.Label(control_frame, text="Azim").pack(side=tk.LEFT)
    
    # Zoom Control
    ttk.Scale(control_frame, from_=5, to=15, orient=tk.HORIZONTAL,
             command=lambda v: update_view(dist=float(v))).pack(side=tk.LEFT, padx=5)
    ttk.Label(control_frame, text="Zoom").pack(side=tk.LEFT)

    # Pack elements
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    return window

# --- OLD FEATURE: Static Density Surfaces ---
def plot_hue_sat_density_surfaces(hue_min_raw, sat_min_raw, hue_max_raw, sat_max_raw):
    fig = plt.figure(figsize=(14, 6))
    
    # Filter out 0 values for accurate density representation
    hmin = hue_min_raw.flatten()
    smin = sat_min_raw.flatten()
    valid_min = (hmin > 0) & (smin > 0)
    hmin_filtered = hmin[valid_min]
    smin_filtered = smin[valid_min]

    hmax = hue_max_raw.flatten()
    smax = sat_max_raw.flatten()
    valid_max = (hmax > 0) & (smax > 0)
    hmax_filtered = hmax[valid_max]
    smax_filtered = smax[valid_max]

    # Min values density surface
    ax1 = fig.add_subplot(121, projection='3d')
    if hmin_filtered.size > 0 and smin_filtered.size > 0:
        hist1, xedges1, yedges1 = np.histogram2d(hmin_filtered, smin_filtered,
                                                 bins=50, range=[[90, 130], [0, 255]])
        X1, Y1 = np.meshgrid(xedges1[:-1], yedges1[:-1])
        ax1.plot_surface(X1, Y1, hist1.T, cmap='viridis',
                         rstride=1, cstride=1, alpha=0.8)
    else:
        ax1.text2D(0.5, 0.5, "No valid min blue pixels.", transform=ax1.transAxes, ha='center')
    ax1.set_title('Min Hue vs Min Saturation Density')
    ax1.set_xlabel('Hue (min)')
    ax1.set_ylabel('Saturation (min)')
    ax1.set_zlabel('Grid Count')

    # Max values density surface
    ax2 = fig.add_subplot(122, projection='3d')
    if hmax_filtered.size > 0 and smax_filtered.size > 0:
        hist2, xedges2, yedges2 = np.histogram2d(hmax_filtered, smax_filtered,
                                                 bins=50, range=[[90, 130], [0, 255]])
        X2, Y2 = np.meshgrid(xedges2[:-1], yedges2[:-1])
        ax2.plot_surface(X2, Y2, hist2.T, cmap='plasma',
                         rstride=1, cstride=1, alpha=0.8)
    else:
        ax2.text2D(0.5, 0.5, "No valid max blue pixels.", transform=ax2.transAxes, ha='center')
    ax2.set_title('Max Hue vs Max Saturation Density')
    ax2.set_xlabel('Hue (max)')
    ax2.set_ylabel('Saturation (max)')
    ax2.set_zlabel('Grid Count')
    plt.tight_layout()
    plt.show()


class BlueInkAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title('Blue Ink Analysis with Advanced Curve Fitting')
        self.root.geometry('1400x900')
        self.image_path = None
        self.blue_hues = None
        self.hue_min = None
        self.hue_max = None
        self.sat_min = None
        self.sat_max = None
        
        # Initial HSV range values
        self.h_lower_val = tk.IntVar(value=90)
        self.s_lower_val = tk.IntVar(value=50)
        self.v_lower_val = tk.IntVar(value=50)
        self.h_upper_val = tk.IntVar(value=130)
        self.s_upper_val = tk.IntVar(value=255)
        self.v_upper_val = tk.IntVar(value=255)
        
        # Grid size
        self.grid_size_val = tk.IntVar(value=100)

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

        # HSV Range Controls
        hsv_frame = ttk.LabelFrame(control_frame, text="Blue Ink HSV Range", padding=10)
        hsv_frame.pack(fill=tk.X, pady=(0, 10))

        # Hue
        ttk.Label(hsv_frame, text="Hue Lower:").grid(row=0, column=0, sticky='w')
        self.h_lower_slider = ttk.Scale(hsv_frame, from_=0, to=179, orient=tk.HORIZONTAL,
                                        variable=self.h_lower_val, length=150)
        self.h_lower_slider.grid(row=0, column=1, sticky='ew')
        ttk.Label(hsv_frame, textvariable=self.h_lower_val).grid(row=0, column=2, sticky='w')

        ttk.Label(hsv_frame, text="Hue Upper:").grid(row=1, column=0, sticky='w')
        self.h_upper_slider = ttk.Scale(hsv_frame, from_=0, to=179, orient=tk.HORIZONTAL,
                                        variable=self.h_upper_val, length=150)
        self.h_upper_slider.grid(row=1, column=1, sticky='ew')
        ttk.Label(hsv_frame, textvariable=self.h_upper_val).grid(row=1, column=2, sticky='w')

        # Saturation
        ttk.Label(hsv_frame, text="Sat Lower:").grid(row=2, column=0, sticky='w')
        self.s_lower_slider = ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        variable=self.s_lower_val, length=150)
        self.s_lower_slider.grid(row=2, column=1, sticky='ew')
        ttk.Label(hsv_frame, textvariable=self.s_lower_val).grid(row=2, column=2, sticky='w')

        ttk.Label(hsv_frame, text="Sat Upper:").grid(row=3, column=0, sticky='w')
        self.s_upper_slider = ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        variable=self.s_upper_val, length=150)
        self.s_upper_slider.grid(row=3, column=1, sticky='ew')
        ttk.Label(hsv_frame, textvariable=self.s_upper_val).grid(row=3, column=2, sticky='w')

        # Value
        ttk.Label(hsv_frame, text="Val Lower:").grid(row=4, column=0, sticky='w')
        self.v_lower_slider = ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        variable=self.v_lower_val, length=150)
        self.v_lower_slider.grid(row=4, column=1, sticky='ew')
        ttk.Label(hsv_frame, textvariable=self.v_lower_val).grid(row=4, column=2, sticky='w')

        ttk.Label(hsv_frame, text="Val Upper:").grid(row=5, column=0, sticky='w')
        self.v_upper_slider = ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        variable=self.v_upper_val, length=150)
        self.v_upper_slider.grid(row=5, column=1, sticky='ew')
        ttk.Label(hsv_frame, textvariable=self.v_upper_val).grid(row=5, column=2, sticky='w')
        
        # Grid Size Control
        grid_frame = ttk.LabelFrame(control_frame, text="Grid Settings", padding=10)
        grid_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(grid_frame, text="Grid Size:").grid(row=0, column=0, sticky='w')
        self.grid_size_slider = ttk.Scale(grid_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                                          variable=self.grid_size_val, length=150)
        self.grid_size_slider.grid(row=0, column=1, sticky='ew')
        ttk.Label(grid_frame, textvariable=self.grid_size_val).grid(row=0, column=2, sticky='w')


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

        self.density3d_button = ttk.Button(process_frame,
                                           text='Show Hue-Sat Density (Static)',
                                           command=self.show_hue_sat_density,
                                           state='disabled')
        self.density3d_button.pack(pady=5)

        self.min_plot_btn = ttk.Button(process_frame, text='Plot Min Hue-Sat (Interactive 3D)',
                                       command=lambda: self.show_density_plot('Min'),
                                       state='disabled')
        self.min_plot_btn.pack(side=tk.LEFT, padx=5, expand=True)

        self.max_plot_btn = ttk.Button(process_frame, text='Plot Max Hue-Sat (Interactive 3D)',
                                       command=lambda: self.show_density_plot('Max'),
                                       state='disabled')
        self.max_plot_btn.pack(side=tk.LEFT, padx=5, expand=True)

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
        
        # Update global HSV range based on slider values
        global LOWER_BLUE, UPPER_BLUE
        LOWER_BLUE = np.array([self.h_lower_val.get(), self.s_lower_val.get(), self.v_lower_val.get()])
        UPPER_BLUE = np.array([self.h_upper_val.get(), self.s_upper_val.get(), self.v_upper_val.get()])

        image = cv2.imread(self.image_path)
        
        # Pass updated HSV range and grid size to the extraction function
        self.hue_min, self.hue_max, self.sat_min, self.sat_max, self.blue_hues = \
            extract_blue_ink_stats(image, grid_size=self.grid_size_val.get(), 
                                   lower_blue_hsv=LOWER_BLUE, upper_blue_hsv=UPPER_BLUE)
        
        for btn in self.fit_buttons.values():
            btn.configure(state='normal')
        self.surface_button.configure(state='normal')
        self.density3d_button.configure(state='normal')
        self.min_plot_btn.configure(state='normal')
        self.max_plot_btn.configure(state='normal')
        self.update_info()
        self.plot_line_histogram('Raw')

    def update_info(self):
        if self.blue_hues is None or len(self.blue_hues) == 0:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, "No blue ink detected or processed yet.")
            return
        
        blue_regions = get_nonzero_blue_regions(self.hue_min, self.hue_max, self.sat_min, self.sat_max)
        
        # Calculate overall min/max hue and saturation from all blue pixels
        overall_min_hue = np.min(self.blue_hues) if len(self.blue_hues) > 0 else 0
        overall_max_hue = np.max(self.blue_hues) if len(self.blue_hues) > 0 else 0
        
        # To get overall saturation min/max, we'd need to collect all blue saturations.
        # For simplicity, we'll indicate current HSV code range.
        
        info_text = f"""Blue Ink Analysis Results:
Current HSV Thresholds:
  Hue: {LOWER_BLUE[0]} - {UPPER_BLUE[0]}
  Saturation: {LOWER_BLUE[1]} - {UPPER_BLUE[1]}
  Value: {LOWER_BLUE[2]} - {UPPER_BLUE[2]}
Grid Size: {self.grid_size_val.get()}x{self.grid_size_val.get()}

Total blue pixels detected (based on current HSV thresholds): {len(self.blue_hues)}
Overall Hue range of detected blue pixels: {overall_min_hue:.1f} - {overall_max_hue:.1f}
Mean hue of detected blue pixels: {np.mean(self.blue_hues):.1f}
Std deviation of detected blue pixels: {np.std(self.blue_hues):.1f}

Grid Analysis (showing first 10 regions if many):
Non-zero blue regions detected: {len(blue_regions)}
"""
        for i, region in enumerate(blue_regions):
            if i >= 10: # Limit display for brevity
                info_text += "...\n"
                break
            info_text += f"Grid {region['grid']}: Hmin={region['hue_min']}, Hmax={region['hue_max']}, Smin={region['sat_min']}, Smax={region['sat_max']}\n"
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)

    def plot_line_histogram(self, fit_type):
        if self.blue_hues is None or len(self.blue_hues) == 0:
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No blue ink data to plot. Adjust HSV range or upload image.", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title('Blue Ink Hue Distribution')
            canvas = FigureCanvasTkAgg(fig, self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define histogram bins based on the current detected hue range, or a default.
        # Using a more dynamic binning or fixed 0-180 could show wider distribution if needed.
        # For now, sticking to 90-130 as per original script, but can be made flexible.
        hist_counts, bin_edges = np.histogram(self.blue_hues, bins=60, range=(0, 179)) 
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax.plot(bin_centers, hist_counts, 'o-', linewidth=1, markersize=3,
                color='blue', alpha=0.7, label='Blue Ink Distribution')
        
        if fit_type != 'Raw':
            x_fit, y_fit, extrema = fit_curve(bin_centers, hist_counts, fit_type)
            
            # Only plot if y_fit has meaningful data (not all zeros from a failed fit)
            if not np.all(y_fit == 0):
                ax.plot(x_fit, y_fit, '-', linewidth=2, color='red', alpha=0.8,
                        label=f'{fit_type} Fit')
                if extrema and 'peaks' in extrema and len(extrema['peaks']) > 0:
                    peaks = extrema['peaks']
                    ax.plot(x_fit[peaks], y_fit[peaks], '^', markersize=8,
                            color='green', label=f'Maxima ({len(peaks)})')
                    for peak in peaks:
                        ax.annotate(f'Max: {x_fit[peak]:.1f}',
                                    xy=(x_fit[peak], y_fit[peak]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, ha='left')
                if extrema and 'valleys' in extrema and len(extrema['valleys']) > 0:
                    valleys = extrema['valleys']
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
        if hasattr(self, 'hue_min') and self.hue_min is not None:
            plot_surface(self.hue_min, self.hue_max, self.sat_min, self.sat_max, title_prefix='Blue Ink')
        else:
            tk.messagebox.showinfo("No Data", "Please process an image first to generate surface plots.")

    def show_hue_sat_density(self):
        if hasattr(self, 'hue_min') and self.hue_min is not None:
            plot_hue_sat_density_surfaces(self.hue_min, self.sat_min, self.hue_max, self.sat_max)
        else:
            tk.messagebox.showinfo("No Data", "Please process an image first to generate density plots.")

    def show_density_plot(self, plot_type):
        if hasattr(self, 'hue_min') and self.hue_min is not None:
            if plot_type == 'Min':
                create_3d_plot_window(self.root, self.hue_min, self.sat_min, 'Min')
            else:
                create_3d_plot_window(self.root, self.hue_max, self.sat_max, 'Max')
        else:
            tk.messagebox.showinfo("No Data", "Please process an image first to generate interactive 3D plots.")

if __name__ == '__main__':
    root = tk.Tk()
    app = BlueInkAnalyzer(root)
    root.mainloop()
