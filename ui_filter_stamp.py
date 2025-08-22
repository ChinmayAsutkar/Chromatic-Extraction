import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QFileDialog, QLabel, QHBoxLayout, QGridLayout,
                           QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Hue range lookup table (0-180, ranges of 20)
# Format: (range_start, range_end): (lower_bound, upper_bound)
# Each range is 20 units wide (0-19, 20-39, etc.)
# The bounds are in HSV format: (H, S, V)
# OpenCV HSV ranges:
# H: 0-180 (not 0-360)
# S: 0-255
# V: 0-255
HUE_RANGE_MASKS = {
    (0, 19): ((110,50,0), (170,255,255)),    # Red-Orange
    (20, 39): ((110,50,0), (170,255,255)),  # Orange-Yellow
    (40, 59): ((110,50,0), (170,255,255)),  # Yellow
    (60, 79): ((82,25,0), (170,255,255)),  # Yellow-Green 1
    (80, 99): ((110,50,0), (170,255,255)),  # Green
    (100, 119): ((110,50,0), (170,255,255)), # Green-Cyan
    (120, 139): ((130,50,0), (170,255,255)), # Cyan
    (140, 159): ((130,50,0), (170,255,255)), # Cyan-Blue
    (160, 180): ((110,107,99), (167,255,255)), # Blue
}

class ImageMatcherUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.template_image = None
        self.search_image = None
        self.filtered_image = None
        self.MIN_MATCH_COUNT = 10
        self.current_fit_type = 'polynomial'  # Default fit type
        
    def initUI(self):
        self.setWindowTitle('Image Matcher')
        self.setGeometry(100, 100, 1800, 1000)  # Increased window size

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create buttons
        self.template_btn = QPushButton('Select Template Image', self)
        self.template_btn.clicked.connect(self.load_template)
        button_layout.addWidget(self.template_btn)

        self.search_btn = QPushButton('Select Search Image', self)
        self.search_btn.clicked.connect(self.load_search)
        button_layout.addWidget(self.search_btn)

        self.process_btn = QPushButton('Process Images', self)
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)

        # Add curve fit type selector
        self.fit_type_combo = QComboBox(self)
        self.fit_type_combo.addItems(['polynomial', 'gaussian', 'spline', 'exponential', 'power'])
        self.fit_type_combo.currentTextChanged.connect(self.change_fit_type)
        button_layout.addWidget(self.fit_type_combo)

        main_layout.addLayout(button_layout)

        # Create grid layout for images and histograms
        grid_layout = QGridLayout()
        
        # Create labels for input images
        self.template_label = QLabel('Template Image')
        self.template_label.setAlignment(Qt.AlignCenter)
        self.template_label.setMinimumSize(300, 300)
        self.template_label.setStyleSheet("border: 1px solid black;")
        grid_layout.addWidget(self.template_label, 0, 0)

        self.search_label = QLabel('Search Image')
        self.search_label.setAlignment(Qt.AlignCenter)
        self.search_label.setMinimumSize(300, 300)
        self.search_label.setStyleSheet("border: 1px solid black;")
        grid_layout.addWidget(self.search_label, 0, 1)

        # Create matplotlib figures for histograms
        self.template_hist_figure = Figure(figsize=(4, 2))
        self.template_hist_canvas = FigureCanvas(self.template_hist_figure)
        grid_layout.addWidget(self.template_hist_canvas, 1, 0)

        self.search_hist_figure = Figure(figsize=(4, 2))
        self.search_hist_canvas = FigureCanvas(self.search_hist_figure)
        grid_layout.addWidget(self.search_hist_canvas, 1, 1)

        # Result label
        self.result_label = QLabel('Result')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(800, 650)
        self.result_label.setStyleSheet("border: 1px solid black;")
        grid_layout.addWidget(self.result_label, 0, 2, 2, 1)

        main_layout.addLayout(grid_layout)

    def change_fit_type(self, fit_type):
        self.current_fit_type = fit_type
        if self.template_image is not None:
            self.plot_hue_histogram(self.template_image, self.template_hist_figure, self.template_hist_canvas)
        if self.search_image is not None:
            self.plot_hue_histogram(self.search_image, self.search_hist_figure, self.search_hist_canvas)

    def fit_curve(self, x, y, fit_type):
        # Filter out extreme values
        mask = (y >= 10) & (y <= 100000)
        if np.sum(mask) < 3:  # If too few points remain, use all points
            mask = np.ones_like(y, dtype=bool)
        
        x_filtered = x[mask]
        y_filtered = y[mask]
        
        if fit_type == 'polynomial':
            # Polynomial fit with increased degree (8)
            coeffs = np.polyfit(x_filtered, y_filtered, 8)
            return np.polyval(coeffs, x)
        elif fit_type == 'gaussian':
            # Gaussian fit
            def gaussian(x, a, b, c):
                return a * np.exp(-(x - b)**2 / (2 * c**2))
            try:
                popt, _ = curve_fit(gaussian, x_filtered, y_filtered, p0=[max(y_filtered), np.argmax(y_filtered), 30])
                return gaussian(x, *popt)
            except:
                return y
        elif fit_type == 'spline':
            # Spline fit with interpolation
            from scipy.interpolate import UnivariateSpline
            # Use interpolation (s=0) for exact fit to filtered points
            spline = UnivariateSpline(x_filtered, y_filtered, k=3, s=0)
            return spline(x)
        elif fit_type == 'exponential':
            # Exponential fit
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            try:
                popt, _ = curve_fit(exp_func, x_filtered, y_filtered, p0=[max(y_filtered), 0.01, 0])
                return exp_func(x, *popt)
            except:
                return y
        elif fit_type == 'power':
            # Power law fit
            def power_func(x, a, b):
                return a * x**b
            try:
                popt, _ = curve_fit(power_func, x_filtered, y_filtered, p0=[max(y_filtered), -1])
                return power_func(x, *popt)
            except:
                return y
        return y

    def plot_hue_histogram(self, image, figure, canvas):
        if image is None:
            return

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for hue channel
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_values = hist.flatten()
        
        # Group histogram values in intervals of 4
        n_groups = 45  # 180/4 = 45 groups
        grouped_hist = np.zeros(n_groups)
        for i in range(n_groups):
            start_idx = i * 4
            end_idx = start_idx + 4
            grouped_hist[i] = np.mean(hist_values[start_idx:end_idx])
        
        # Clear the figure
        figure.clear()
        
        # Create subplot
        ax = figure.add_subplot(111)
        
        # Create color map for grouped bars
        colors = []
        for i in range(n_groups):
            # Use the middle hue value of each group for the color
            middle_hue = i * 4 + 2
            hsv_color = np.uint8([[[middle_hue, 255, 255]]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            color = rgb_color / 255.0
            colors.append(color)
        
        # Plot histogram with grouped bars
        x = np.arange(n_groups)
        bars = ax.bar(x, grouped_hist, color=colors, alpha=0.7)
        
        # Fit curve to grouped data
        y_fit = self.fit_curve(x, grouped_hist, self.current_fit_type)
        ax.plot(x, y_fit, 'k-', linewidth=2, label=f'{self.current_fit_type} fit')
        
        # Find local maxima only for significant peaks (between 10 and 100000)
        significant_mask = (grouped_hist >= 10) & (grouped_hist <= 100000)
        if np.any(significant_mask):
            peaks, _ = find_peaks(y_fit, height=10, distance=2)  # Reduced distance for grouped data
            ax.plot(x[peaks], y_fit[peaks], "rx", markersize=10, label='Local Maxima')
        
        # Set labels and title
        ax.set_xlabel('Hue Group (4 values per group)')
        ax.set_ylabel('Average Frequency (log scale)')
        ax.set_title(f'Hue Distribution with {self.current_fit_type.capitalize()} Fit')
        
        # Set x-axis ticks for groups
        ax.set_xticks(range(0, n_groups, 5))  # Show every 5th group
        ax.set_xticklabels([f'{i*4}-{i*4+3}' for i in range(0, n_groups, 5)])
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.1)
        
        # Add legend
        ax.legend()
        
        # Update canvas
        canvas.draw()

    def display_image(self, image, label):
        if image is None:
            label.setText("No image loaded")
            return

        if len(image.shape) == 2:  # Grayscale image
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color image
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def apply_color_filter(self, image):
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Sample 1000 random points from the image
        height, width = image.shape[:2]
        random_points = np.random.randint(0, height * width, 1000)
        y_coords = random_points // width
        x_coords = random_points % width
        
        # Get colors of sampled points
        sampled_colors = image[y_coords, x_coords]
        
        # Calculate median color
        median_color = np.median(sampled_colors, axis=0).astype(np.uint8)
        
        # Convert median color to HSV to get the hue
        median_hsv = cv2.cvtColor(np.uint8([[median_color]]), cv2.COLOR_BGR2HSV)[0][0]
        median_hue = median_hsv[0]
        
        # Find the appropriate range for the median hue
        hue_range = None
        for (range_start, range_end) in HUE_RANGE_MASKS.keys():
            if range_start <= median_hue <= range_end:
                hue_range = (range_start, range_end)
                break
        
        # If no range found, use a default range
        if hue_range is None:
            hue_range = (0, 19)  # Default to first range
        
        # Get the mask bounds for this hue range
        lower_bound, upper_bound = HUE_RANGE_MASKS[hue_range]
        
        # Create and apply the mask
        mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
        mask_inv = cv2.bitwise_not(mask)

        # Apply mask to get only regions matching the hue range
        filtered = cv2.bitwise_and(image, image, mask=mask)
        
        # Replace non-matching regions with median color
        filtered[mask!=255] = median_color
        
        return filtered

    def load_template(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Template Image')
        if fname:
            self.template_image = cv2.imread(fname)
            if self.template_image is not None:
                self.template_image = self.apply_color_filter(self.template_image)
                self.display_image(self.template_image, self.template_label)
                self.plot_hue_histogram(self.template_image, self.template_hist_figure, self.template_hist_canvas)
                self.check_can_process()

    def load_search(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Search Image')
        if fname:
            self.search_image = cv2.imread(fname)
            if self.search_image is not None:
                self.search_image = self.apply_color_filter(self.search_image)
                self.display_image(self.search_image, self.search_label)
                self.plot_hue_histogram(self.search_image, self.search_hist_figure, self.search_hist_canvas)
                self.check_can_process()

    def check_can_process(self):
        if self.template_image is not None and self.search_image is not None:
            self.process_btn.setEnabled(True)

    def process_images(self):
        if self.template_image is None or self.search_image is None:
            return

        # Convert both filtered images to grayscale for SIFT matching
        template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
        search_gray = cv2.cvtColor(self.search_image, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(template_gray, None)
        kp2, des2 = sift.detectAndCompute(search_gray, None)

        # Check if descriptors were found
        if des1 is None or des2 is None:
            self.filtered_image = self.search_image
            self.display_image(self.filtered_image, self.result_label)
            return

        # Convert descriptors to float32 for FLANN
        des1 = np.float32(des1)
        des2 = np.float32(des2)

        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=150)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            # Find matches
            matches = flann.knnMatch(des1, des2, k=2)

            # Apply Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # Create a copy of search image for drawing
            result = self.search_image.copy()

            if len(good) > self.MIN_MATCH_COUNT:
                # Get matching points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                # Get corners of template image
                h, w = template_gray.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                
                # Transform corners to search image coordinates
                dst = cv2.perspectiveTransform(pts, M)

                # Draw rectangle around matched area
                result = cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                # Calculate convex hull areas
                # For template image
                template_inliers = src_pts[mask.ravel() == 1]
                if len(template_inliers) > 2:
                    template_hull = cv2.convexHull(template_inliers)
                    template_area = cv2.contourArea(template_hull)
                    template_percentage = (template_area / (w * h)) * 100
                else:
                    template_percentage = 0

                # For search image
                search_inliers = dst_pts[mask.ravel() == 1]
                if len(search_inliers) > 2:
                    search_hull = cv2.convexHull(search_inliers)
                    search_area = cv2.contourArea(search_hull)
                    search_percentage = (search_area / (search_gray.shape[0] * search_gray.shape[1])) * 100
                else:
                    search_percentage = 0

                # Determine if matched based on percentage difference
                percentage_diff = abs(template_percentage - search_percentage)
                ratio = template_percentage / search_percentage if search_percentage > 0 else 0
                
                # Match if:
                # 1. At least 10 inlier points
                # 2. Percentage difference is less than or equal to 10
                # 3. Ratio between template and search areas is between 0.75 and 1.25
                match_status = "Matched" if (len(template_inliers) >= 10 and 
                                           percentage_diff <= 10 and 
                                           0.65 <= ratio <= 1.5) else "Not Matched"

                # Draw matches
                draw_params = dict(matchColor=(0, 255, 0),
                                 singlePointColor=None,
                                 matchesMask=matchesMask,
                                 flags=2)
                
                # Create a combined image showing matches
                matches_img = cv2.drawMatches(template_gray, kp1, search_gray, kp2, good, None, **draw_params)
                
                # Draw convex hulls on the matches image
                if len(template_inliers) > 2:
                    # Draw template convex hull
                    template_hull_points = np.int32(template_hull.reshape(-1, 2))
                    for i in range(len(template_hull_points)):
                        pt1 = template_hull_points[i]
                        pt2 = template_hull_points[(i+1) % len(template_hull_points)]
                        cv2.line(matches_img, tuple(pt1), tuple(pt2), (255, 0, 0), 2)
                
                if len(search_inliers) > 2:
                    # Draw search convex hull
                    search_hull_points = np.int32(search_hull.reshape(-1, 2))
                    # Offset the points to the right side of the image
                    offset = w
                    for i in range(len(search_hull_points)):
                        pt1 = (search_hull_points[i][0] + offset, search_hull_points[i][1])
                        pt2 = (search_hull_points[(i+1) % len(search_hull_points)][0] + offset, 
                              search_hull_points[(i+1) % len(search_hull_points)][1])
                        cv2.line(matches_img, pt1, pt2, (255, 0, 0), 2)
                
                # Get the matched hue range
                hsv = cv2.cvtColor(self.search_image, cv2.COLOR_BGR2HSV)
                height, width = self.search_image.shape[:2]
                random_points = np.random.randint(0, height * width, 1000)
                y_coords = random_points // width
                x_coords = random_points % width
                sampled_colors = self.search_image[y_coords, x_coords]
                median_color = np.median(sampled_colors, axis=0).astype(np.uint8)
                median_hsv = cv2.cvtColor(np.uint8([[median_color]]), cv2.COLOR_BGR2HSV)[0][0]
                median_hue = median_hsv[0]
                
                # Find the matched hue range
                matched_range = None
                for (range_start, range_end) in HUE_RANGE_MASKS.keys():
                    if range_start <= median_hue <= range_end:
                        matched_range = f"Hue {range_start}-{range_end}"
                        break
                
                if matched_range is None:
                    matched_range = "Hue 0-19"  # Default range
                
                # Add text to the matches image
                text_color = (0, 255, 0) if match_status == "Matched" else (0, 0, 255)
                cv2.putText(matches_img, f"Template Area: {template_percentage:.2f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(matches_img, f"Search Area: {search_percentage:.2f}%", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(matches_img, f"Inlier Points: {len(template_inliers)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(matches_img, f"Status: {match_status}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(matches_img, f"Matched Range: {matched_range}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                # Display the result with matches
                self.filtered_image = matches_img
                self.display_image(self.filtered_image, self.result_label)
            else:
                # If not enough matches found, just show the search image
                self.filtered_image = result
                self.display_image(self.filtered_image, self.result_label)
        except cv2.error as e:
            print(f"Error in matching: {e}")
            # If matching fails, show the search image
            self.filtered_image = self.search_image
            self.display_image(self.filtered_image, self.result_label)

def main():
    app = QApplication(sys.argv)
    ex = ImageMatcherUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
