import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


def deskew_image(img):
    """Deskew image using moments."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # Only deskew if the angle is significant (e.g., more than 1 degree)
    if abs(angle) > 1:
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed
    else:
        # Return original image if no significant skew
        return img


def normalize_image(img):
    """Normalize the image (histogram equalization on the V channel of HSV)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return normalized


def remove_shadow(img):
    """Remove shadows using morphology and normalization."""
    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((15, 15), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        normed = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        result_planes.append(normed)
    result = cv2.merge(result_planes)
    return result


def cvimg2qpixmap(img):
    """Convert OpenCV image to QPixmap."""
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)

def clahe_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced


def adaptive_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)


from PIL import Image, ExifTags

def load_image_with_exif_orientation(filename):
    # Open image with Pillow and auto-orient
    pil_img = Image.open(filename)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = dict(pil_img._getexif().items())

        if exif[orientation] == 3:
            pil_img = pil_img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            pil_img = pil_img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            pil_img = pil_img.rotate(90, expand=True)
    except Exception as e:
        pass  # If no EXIF or orientation, keep original

    # Convert to OpenCV format
    cv_img = np.array(pil_img)
    if cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGR)
    else:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


def show_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    # Convert edge image to 3 channels for display
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edge_bgr

def has_shadow(img, thresh=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Estimate background
    blurred = cv2.medianBlur(gray, 51)
    stddev = np.std(blurred)
    return stddev > thresh

class SlipPreprocessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slip Preprocessing Demo")
        # Increase window size
        self.setGeometry(100, 100, 1600, 1000)  # Increased window size
        
        # Create labels with larger minimum sizes
        self.original_label = QLabel("Original Image")
        self.original_label.setMinimumSize(600, 800)  # Increased size
        self.original_label.setStyleSheet("border: 1px solid black;")
        
        self.processed_label = QLabel("Processed Image")
        self.processed_label.setMinimumSize(600, 800)  # Increased size
        self.processed_label.setStyleSheet("border: 1px solid black;")
        
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setMinimumHeight(40)  # Make button bigger
        self.open_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Create layouts
        layout = QHBoxLayout()
        layout.addWidget(self.original_label)
        layout.addWidget(self.processed_label)
        layout.setSpacing(20)  # Add spacing between images

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(self.open_button, alignment=Qt.AlignCenter)
        main_layout.setSpacing(20)  # Add spacing between elements
        self.setLayout(main_layout)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', "Image files (*.jpg *.jpeg *.png *.bmp)")
        if fname:
            img = load_image_with_exif_orientation(fname)
            # Increase resize dimensions
            img_resized = cv2.resize(img, (800, 1000), interpolation=cv2.INTER_AREA)

            if has_shadow(img_resized):
                result_image = remove_shadow(img_resized)
                print("shadow removed")
            else:
                result_image = (img_resized)
                print("NO shadow removed")

            # Display larger images
            self.original_label.setPixmap(cvimg2qpixmap(img_resized).scaled(
                600, 800,  # Increased display size
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.processed_label.setPixmap(cvimg2qpixmap(result_image).scaled(
                600, 800,  # Increased display size
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SlipPreprocessor()
    window.show()
    sys.exit(app.exec_())