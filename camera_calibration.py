"""
Camera calibration using OpenCV and a series of chessboard test images
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pickle

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store all object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Find the camera matrix and dist coeffs using the provided chessboard images
camera_cal_images =  glob.glob('camera_cal/*.jpg')
chessboard_axes = (9, 6)

for idx, fname in enumerate(camera_cal_images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners using cv2
    ret, corners = cv2.findChessboardCorners(gray, chessboard_axes, None)

    # Add detected corners to objpoints, imgpoints
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# Find image size using a test image
test_image = cv2.imread('camera_cal/calibration3.jpg')
img_size = (test_image.shape[1], test_image.shape[0])

# Calibrate the camera using cv2.calibrateCamera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Test the undistortion on the test image
dst = cv2.undistort(test_image, mtx, dist, None, mtx)

# Save the camera matrix and distortion coefficients for use in the pipeline
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("dist_pickle.p", "wb"))

# Save the dist and undist images to writeup_images folder
cv2.imwrite("writeup_images/camera_cal_original.jpg", test_image)
cv2.imwrite("writeup_images/camera_cal_undist.jpg", dst)

# Visualise the result on the test image
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.imshow(test_image)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(dst)
# ax2.set_title('Undistorted Image', fontsize=30)
# plt.show()