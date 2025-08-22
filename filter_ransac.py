#! /usr/bin/python3

import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QFileDialog, QLabel, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

MIN_MATCH_COUNT = 10
denoise=0

img1_path = 'img1.jpeg'
img2_path = 'img2.jpeg'

if(denoise):
    image = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
    img1 = cv2.fastNlMeansDenoising(image, None, 20, 7, 21)

    image = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)
    img2 = cv2.fastNlMeansDenoising(image, None, 20, 7, 21)
else:
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Could not load one or both images. Check the file paths!")
    exit()
else:
    print("image loaded correctly")


#blue filter
frame = cv2.imread(img2_path)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
# Threshold of blue in HSV space
# lower_blue = np.array([60, 35, 20])
# upper_blue = np.array([180, 255, 150])

# WORKING BLUE RANGE
lower_blue = np.array([100, 25, 80])
upper_blue = np.array([115, 255, 150])

# refined blue range
# lower_blue = np.array([90, 25, 10])
# upper_blue = np.array([128, 255, 200])


# lower_blue = np.array([110,50,50])
# upper_blue = np.array([130,255,255])

#handle white
sensitivity = 15
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])

# lower_blue = np.array([90, 25, 10])
# upper_blue = np.array([158, 255, 255])




lower=(0,170,215)
upper=(70,255,255)

# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_blue, upper_blue)

#mask = cv2.inRange(frame, lower, upper)
#mask = 255 - mask
mask_inv = cv2.bitwise_not(mask)

# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
frame = cv2.bitwise_and(frame, frame, mask = mask)
result=frame.copy()
#result[mask!=255] = (180, 180, 180)
result[mask!=255] = (150, 150, 150)

#result = cv2.bitwise_xor(result, white_mask)

# cv2.imshow('frame', frame)
# cv2.waitKey(0)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# cv2.imshow('result', result)
# cv2.waitKey(0)

# plt.imshow(result, 'gray'),plt.show()
  
# cv2.waitKey(0)

#cv2.destroyAllWindows()

#img2 = result

# //comenting this part
img2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# //com
plt.imshow(img2, 'gray'),plt.show()

#blue filter END


# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


if des1 is None:
    print("Could not find SIFT features in one ")
    exit()
if des2 is None:
    print("Could not find SIFT features in two ")
    exit()


des1 = np.float32(des1)
des2 = np.float32(des2)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 150)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        (x, y) = kp1[m.queryIdx].pt
#        print(x, y)


#print ("Matches found - %d/%d" % (len(good),len(matches)))
numInliner=0

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    color_line = (0,255,0)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,color_line,3, cv2.LINE_AA)
    for x in matchesMask:
        if(x):
            numInliner=1+numInliner

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    print(len(good))
    matchesMask = None


print ("Inlier Matches found - %d/%d" % (numInliner,len(matches)))

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()

class ImageMatcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.img1 = None
        self.img2 = None
        self.MIN_MATCH_COUNT = 10

    def initUI(self):
        self.setWindowTitle('Image Matcher')
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create buttons
        self.img1_btn = QPushButton('Select First Image', self)
        self.img1_btn.clicked.connect(self.load_first_image)
        button_layout.addWidget(self.img1_btn)

        self.img2_btn = QPushButton('Select Second Image', self)
        self.img2_btn.clicked.connect(self.load_second_image)
        button_layout.addWidget(self.img2_btn)

        self.process_btn = QPushButton('Process Images', self)
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)

        main_layout.addLayout(button_layout)

        # Create image display layout
        image_layout = QHBoxLayout()
        
        # Create labels for images
        self.img1_label = QLabel('First Image')
        self.img1_label.setAlignment(Qt.AlignCenter)
        self.img1_label.setMinimumSize(400, 400)
        self.img1_label.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.img1_label)

        self.img2_label = QLabel('Second Image')
        self.img2_label.setAlignment(Qt.AlignCenter)
        self.img2_label.setMinimumSize(400, 400)
        self.img2_label.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.img2_label)

        main_layout.addLayout(image_layout)

    def load_first_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select First Image')
        if fname:
            self.img1 = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if self.img1 is not None:
                self.display_image(self.img1, self.img1_label)
                self.check_can_process()

    def load_second_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Second Image')
        if fname:
            self.img2 = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if self.img2 is not None:
                self.display_image(self.img2, self.img2_label)
                self.check_can_process()

    def check_can_process(self):
        if self.img1 is not None and self.img2 is not None:
            self.process_btn.setEnabled(True)

    def display_image(self, image, label):
        if image is None:
            return

        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def process_images(self):
        if self.img1 is None or self.img2 is None:
            return

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)

        if des1 is None or des2 is None:
            print("Could not find SIFT features in one or both images")
            return

        des1 = np.float32(des1)
        des2 = np.float32(des2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=150)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = self.img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw the matches
            draw_params = dict(matchColor=(0, 255, 0),
                             singlePointColor=None,
                             matchesMask=matchesMask,
                             flags=2)

            img3 = cv2.drawMatches(self.img1, kp1, self.img2, kp2, good, None, **draw_params)
            
            # Display the result
            plt.figure(figsize=(12, 8))
            plt.imshow(img3, 'gray')
            plt.show()

        else:
            print(f"Not enough matches are found - {len(good)}/{self.MIN_MATCH_COUNT}")

def main():
    app = QApplication(sys.argv)
    ex = ImageMatcher()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()