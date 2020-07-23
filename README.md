# Advanced Lane Finding Project

The goal of this project is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car, and provide the radius of curvature and vehicle position. 

Find my project on [Github](https://github.com/syedtaqi95/CarND-Advanced-Lane-Lines).

![output_test1]

The project is broken down into the following steps:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[output_test1]: ./output_images/output_test1.jpg "output_test1"
[camera_cal]: ./output_images/camera_cal.jpg "camera_cal"
[undistortion]: ./output_images/undistortion.jpg "undistortion"
[combined_binary]: ./output_images/combined_binary.jpg "combined_binary"
[warped]: ./output_images/warped.jpg "warped"
[sliding_window]: ./output_images/sliding_window.jpg "sliding_window"
[search_around_poly]: ./output_images/search_around_poly.jpg "search_around_poly"
[final_output]: ./output_images/final_output.jpg "final_output"

---

## Rubric Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation. I will visualise each step using test images.

### **Writeup / README**

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point. 

You're reading it!

### **Camera Calibration**

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in `camera_calibration.py`. I referenced the code located [here](https://github.com/syedtaqi95/CarND-Camera-Calibration) for this step.

First I prepared a set of object points (stored in `objp`) such as (0, 0, 0), (1, 0, 0), ... (8, 5, 0). These are the points of the chessboard corners in 3D space for the 9x6 chessboard used in this project. Note that I assumed z=0 so all the test images are on the same plane. When I detected the chessboard corners, I simply appended a copy of `objp` to `objpoints`. The image points (stored in `imgpoints`) are the set of (x, y) points of the detected chessboard corners in pixel space.

I looped through the images located in the *test_images* directory and used OpenCV's `cv2.findChessboardCorners()` function to detect the corners. Once detected, I stored the object and image points in the previously mentioned variables.

I computed the camera matrix and distortion coefficients using OpenCV's `cv2.calibrateCamera()` function. Finally I used the `cv2.undistort()` function on a test image to confirm that distortion correction is working correctly. The following result proved that the correction worked well:

![camera_cal]

I saved the camera matrix and distortion coefficients in `dist_pickle.p` so that I can quickly load it in my main pipeline file.

### **Pipeline (test images)**

The pipeline is wrapped in a function called `process_image()` (line 349).

#### 1. Provide an example of a distortion-corrected image.

This step is in `advanced_lane_pipeline.py` line 357.

![undistortion]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

This step is in the `create_binary_image()` function (line 25).

The function uses both gradient and colour thresholds to create the binary image. For colour thresholding, I converted the undistorted image from RGB to the HLS colour space to extract the S channel. After testing different values, I got a good result (for both white and yellow pixels) by using the following colour thresholds on the S channel:

```python
s_thresh_min = 170
s_thresh_max = 255
```

I also converted the undistorted image to grayscale and applied the Sobel operator in the x direction using the `cv2.Sobel()` function. Combining the two yielded the following output:

![combined_binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This step is in the `perspective_transform()` function (line 58).

The basic principle is fairly simple - given a set of source and destination points, `cv2.getPerspectiveTransform()` computes a transform matrix `M`. I applied this transform matrix to the image to get the warped output using `cv2.warpPerspective()`. Note that I also compute the inverse matrix `Minv` to unwarp the final image back to the original perpective.

I played around with the source and destination points on the provided *straight_lines* images. My goal was to get the warped lines as parallel as possible. I found the following set of points worked best for me:

```python
# Source and destination points
src = np.float32([[195,720], [555,475], [730,475], [1125,720]])
dst = np.float32([[320,720], [320,0], [960,0], [960,720]])
```

Which yielded the following result:

![warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used a `Line()` class to store some useful information about each lane, and created `left_line` and `right_line` objects in my application. See `line.py` for details of my implementation.

This step was a bit more complex so I split it up into the following functions:

- `detect_lane_pixels()` (line 86)
- `sliding_window()` (line 122)
- `search_around_poly()` (line 208)
- `fit_polynomial()` (line 272)
- `sanity_check()` (line 287)

First, if the lane was not detected before (i.e. if it's the first frame or if the sanity check failed), then run the `sliding_window()` function. This function creates a histogram based on the activated pixels from the warped binary image, and uses a sliding window to return the useful pixels for generating a polynomial later. Applying this to our test image:

![sliding_window]

Alternatively, if there was already a polynomial fit from a previous frame, I used the `search_around_poly()` function to search around the previous polynomial. Rather than doing a blind search, this function only searched for activated pixels around a margin using the previously detected polynomial. It is a type of region of interest selection, but more localised compared to the first project. Applying this to our test image:

![search_around_poly]

Once I found the useful activated pixels through either of the two methods, I used the `fit_polynomial()` function to fit a second order polynomial. This uses the `np.polyfit()` function to generate second order polynomial coefficients given a set of points. You can see the polynomial as a yellow line in the previous image.

The final bit of this step is the `sanity_check()` function. While I tried a number of different sanity check implementations, the one I used in the end was very simple and provided a good result on the test images and the project video. It simply checks the lane width at the top and bottom of the warped image, and if the difference between the two is larger than 50 pixels, the sanity check fails. If the sanity check fails, I simply discard the detected polynomial and use the previous  polynomial in `line.currentfit`. If it passes, I save the new polynomial to the `Line()` objects.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This step uses the `find_curvature_and_offset()` function (line 287).

I defined a conversion from pixel space to the real-world world (in metres) as below:

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

For the curvature calculation, I simply used the bottom of the image as the chosen y-value. The calculation of the left and right radii is given by:

```python
left_curverad = (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5 / np.absolute(2*left_fit_cr[0])
right_curverad = (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5 / np.absolute(2*right_fit_cr[0])
```
The output radius is a mean of the left and right radii:

```python
curvature = round(np.mean([left_curverad, right_curverad]), 0)
```

I calculated the vehicle offset and position using the bottom of the image as my chosen reference point. As we assumed the camera is mounted in the centre of the vehicle, the offset is the difference between the centre of the detected lanes and the midpoint of the image. The vehicle position (i.e. left or right of the lane center) was calculated similarly - if the lane centre is to the right of the midpoint, the vehicle is to the left, and vice versa. Again, the values used were converted from pixels to metres.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this is part of the `process_image()` function (lines 378-397).

I simply used OpenCV's `cv2.putText()` function to overlay the curvature and vehicle position information on each frame. This is my final output:

![final_output]

### **Pipeline (video)**

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

My project video output is called *output_project_video.mp4* in the project repositary and has been uploaded to Youtube [here](https://youtu.be/fX5k2I6cmdY).

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The main issues I faced while doing this project were:
- Tuning the parameters for each step took a lot of trial and error. Moreover, what worked for the test images did not always work for the project video, and vice versa.
- Initially I had all my code as a monolithic block of code. Splitting it out into individual functions helped me speed up unit testing.
- Getting the sanity check to work reliably was an issue for me. It kept failing at different points in the video until my final strategy, which is quite simple actually.

My pipeline will likely fail in the following scenarios:
- When the first frame does not generate a 'good' polynomial. This is in frames where the histogram peaks do not correspond to the lanes.
- When there is an obstacle on the lane in front of the ego vehicle.
- In conditions where either one or both of the lane markings are not visible very well, such as dark lighting conditions.

If I were to continue working on this project, I would make the following improvements:
- Change the `src` and `dst` warping points from static pixel positions to dynamic, i.e. based on the width and height of the video. This would allow me to process a video of any frame size.
- Employ a more complex strategy to estimate the lane curvature, rather than just use the bottom of the image as my reference point. Something like calculating the mean value of the curvature at multiple points would provide a more accurate value.
- Detect left and right lanes separately. Currently if one lane detection fails, both detections are rejected. Detections would improve if I detected each lane separately.
- Not sure if this will provide any improvements, but I think using the full image for the histogram (rather than just the bottom half) might improve detections, as we are simply throwing away half the image.
- Apply smoothing to reduce jitter.

In conclusion, I found this project very interesting as it built on the knowledge of the previous project and allowed me to see my algorithm improve iteratively. It also taught me a lot of new computer vision techniques which will be invaluable knowledge for working on self-driving cars!

