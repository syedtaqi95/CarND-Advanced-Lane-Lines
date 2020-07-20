"""
Advanced Lane Detection Pipeline:
- Camera calibration (calculated in camera_calibration.py)
- Apply a distortion correction to raw images.
- Use colour transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pickle

"""
Loading camera calibration data and loading images
"""
# Load the camera matrix and dist coefficients from dist_pickle
# NOTE: These values were calculated in camera_calibration.py
pickle_data = pickle.load(open("dist_pickle.p", "rb"))
mtx = pickle_data["mtx"]
dist = pickle_data["dist"]

# Apply the pipeline on a test image
# TODO: run the pipeline on all the test images
img = cv2.imread("test_images/test1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get image size
img_size = (img.shape[1], img.shape[0])

"""
Undisortion
"""
# Undistort the image using the calculated dist coefficients and camera matrix
undist = cv2.undistort(img_rgb, mtx, dist, None, mtx)

"""
Colour transform and gradient estimation
"""
# Convert to grayscale
undist_gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

# Convert to HLS and extract the S channel
hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]

# Sobel in x
sobelx = cv2.Sobel(undist_gray, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(abs_sobelx*255/np.max(abs_sobelx))

# Gradient threshold
g_thresh_min = 30
g_thresh_max = 120
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= g_thresh_min) & (scaled_sobel <= g_thresh_max)] = 1

# Colour threshold
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

# Combine the binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1

"""
Perspective transform
"""
src_coord = [[195,720], [555,475], [730,475], [1125,720]]
dst_coord = [[320,720], [320,0], [960,0], [960,720]]

src = np.float32(src_coord)
dst = np.float32(dst_coord)
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(combined_binary, M, img_size)




# Plotting for debug purposes
src_coord.append(src_coord[0])
xs1, ys1 = zip(*src_coord)
dst_coord.append(dst_coord[0])
xs2, ys2 = zip(*dst_coord)

f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('combined_binary')
ax1.imshow(combined_binary, cmap = 'gray')
ax1.plot(xs1,ys1)

ax2.set_title('warped')
ax2.imshow(warped, cmap = 'gray')
ax2.plot(xs2,ys2)

plt.tight_layout()
plt.show()
