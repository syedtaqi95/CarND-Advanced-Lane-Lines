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
import pickle
import os
from moviepy.editor import VideoFileClip
from line import Line


"""
Colour transform and gradient estimation
"""
def create_binary_image(undist, g_thresh=(30,120), s_thresh=(170,255)):
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
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= g_thresh[0]) & (scaled_sobel <= g_thresh[1])] = 1

    # Colour threshold
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1

    return combined_binary


"""
Perspective transform
"""
def perspective_transform(combined_binary, img_size):
    # Source and destination points
    src = np.float32([[195,720], [555,475], [730,475], [1125,720]])
    dst = np.float32([[320,720], [320,0], [960,0], [960,720]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(combined_binary, M, img_size)

    return warped, M, Minv


"""
Detect lane pixels
"""
def detect_lane_pixels(warped, img_size=(1280,720),):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped, warped, warped))*255
    
    # If lane has not been detected, use sliding window to detect lane
    if left_line.detected == False | right_line.detected == False:
        leftx, lefty, rightx, righty = sliding_window(warped, out_img, img_size)
        left_line.detected = True
        right_line.detected = True
    # Else, search around the old polynomial
    else:
        leftx, lefty, rightx, righty = search_around_poly(warped, left_line.current_fit, \
            right_line.current_fit, out_img, img_size)

    # Fit the polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = \
        fit_polynomial(img_size, leftx, lefty, rightx, righty)
    
    # Sanity check - only save the values if they make sense
    # Save anyway if it's the first frame
    if (sanity_check(left_fitx, right_fitx) == True)  | (left_line.first_frame == True):
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit
        left_line.current_fitx = left_fitx
        right_line.current_fitx = right_fitx
        left_line.first_frame = False
        left_line.detected = False

    return left_line.current_fitx, right_line.current_fitx, \
        left_line.current_fit, right_line.current_fit


"""
Sliding window using histogram
- used for the first frame
"""
def sliding_window(warped, out_img, img_size=(1280,720)):
    # Create a histogram from the warped image
    bottom_half = warped[img_size[1]//2:,:]
    histogram = np.sum(bottom_half, axis = 0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 60

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img_size[1]//nwindows)
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Convert from a list of lists to a regular list
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Visualization
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255] 

    return leftx, lefty, rightx, righty


"""
Search around the old polynomial
"""
def search_around_poly(warped, left_fit, right_fit, out_img, img_size=(1280,720)):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Find good pixels within the margin of the polynomial
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = \
        fit_polynomial(img_size, leftx, lefty, rightx, righty)

    ## Visualization ##
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    search_area = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    # ax2.plot(left_fitx, ploty, color='yellow')
    # ax2.plot(right_fitx, ploty, color='yellow')
    # ax2.imshow(search_area, cmap = 'gray')

    ## End visualization steps ##
    
    return leftx, lefty, rightx, righty


"""
Fit a polynomial given some points
"""
def fit_polynomial(img_size, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_size[1]-1, img_size[1] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, left_fit, right_fit

"""
Sanity check - works by checking the width of the bottom and top points of the curve
"""
def sanity_check(left_fitx, right_fitx):
    bottom_diff = right_fitx[-1] - left_fitx[-1]
    top_diff = right_fitx[0] - left_fitx[0]

    if np.absolute(bottom_diff - top_diff) <= 50:
        return True
    else:
        return False


"""
Curvature and vehicle position calculation
"""
def find_curvature_and_offset(left_fitx, right_fitx, left_fit, right_fit, img_size=(1280,720)):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit a second order polynomial to pixel positions in each line
    ploty = np.linspace(0, img_size[1]-1, img_size[1] )
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty)

    # Calculation of R_curve
    left_curverad = (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5 / np.absolute(2*left_fit_cr[0])
    right_curverad = (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5 / np.absolute(2*right_fit_cr[0])
    curvature = round(np.mean([left_curverad, right_curverad]), 0)

    # Calculation of vehicle position
    midpoint = img_size[0]//2
    left_lane_centre = left_fitx[-1]
    right_lane_centre = right_fitx[-1]
    lane_centre = right_lane_centre - left_lane_centre
    vehicle_offset = round(np.absolute((midpoint - lane_centre)*xm_per_pix), 2)

    # Check if vehicle is to the left or right of lane centre
    # Assumes the camera is mounted at the centre of the vehicle
    vehicle_position = "right"
    if(lane_centre >= midpoint):
        vehicle_position = "left"

    return curvature, vehicle_position, vehicle_offset


"""
Unwarp the detected lanes back to the original image and plot
"""
def unwarp_image(undist, warped, left_fitx, right_fitx, Minv, img_size=(1280,720)):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, img_size[1]-1, img_size[1] )
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result


"""
Main processing pipeline for each frame
"""
def process_image(image):
    # Convert to RGB to make life easier
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image size
    img_size = (image.shape[1], image.shape[0])

    # Undistort the image using the calculated dist coefficients and camera matrix
    undist = cv2.undistort(img_rgb, mtx, dist, None, mtx)

    # Create the binary image
    combined_binary = create_binary_image(undist)
    
    # Perspective transform
    warped, M, Minv = perspective_transform(combined_binary, img_size)

    # Detect lane pixels
    left_fitx, right_fitx, left_fit, right_fit \
        = detect_lane_pixels(warped, img_size=img_size)

    curvature, vehicle_position, vehicle_offset \
        = find_curvature_and_offset(left_fitx, right_fitx, left_fit, right_fit)

    # Unwarp the image and plot the detected lanes
    result = unwarp_image(undist, warped, left_fitx, right_fitx, Minv, img_size=img_size)
    
    # Overlay the text on the final image
    # Radius of curvature
    cv2.putText(result,
        "Radius of curvature = " + str(curvature) + "(m)",
        (80,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255,255,255),
        2
        )

    # Vehicle position
    cv2.putText(result,
        "Vehicle is " + str(vehicle_offset) + "m " + vehicle_position + " of center",
        (80,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255,255,255),
        2
        )
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':

    f, (ax1,ax2) = plt.subplots(1,2)

    # Load the camera matrix and dist coefficients from dist_pickle
    # These values were calculated in camera_calibration.py
    pickle_data = pickle.load(open("dist_pickle.p", "rb"))
    mtx = pickle_data["mtx"]
    dist = pickle_data["dist"]

    # Create left and right lane Line objects
    left_line = Line()
    right_line = Line()
    
    # Run the pipeline on the project video and save it
    filename = "project_video.mp4"
    video_output = "output_" + filename
    clip = VideoFileClip(filename)
    processed_clip = clip.fl_image(process_image)
    processed_clip.write_videofile(video_output, audio=False)

    # OPTIONAL - Apply the pipeline on the test images and save them
    # test_images = os.listdir("test_images")
    # for idx, filename in enumerate(test_images):
    #     left_line = Line()
    #     right_line = Line()
    #     img = cv2.imread("test_images/" + filename)
    #     output_image = process_image(img)
    #     # Save the file to the output_images folder
    #     cv2.imwrite("output_images/output_"+filename, output_image)


