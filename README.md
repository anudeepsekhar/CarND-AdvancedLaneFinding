# CarND-AdvancedLaneFinding
**The goals / steps of this project are the following:**

**Processing the image:**

     1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
     2. Apply a distortion correction to raw images.
     3. Use color transforms, gradients, etc., to create a thresholded binary image.
     4. Apply a perspective transform to rectify binary image ("birds-eye view").
     5. Detect lane pixels and fit to find the lane boundary.
     6. Sanity check to verify the lines are detected correctly
     7. Determine the curvature of the lane and vehicle position with respect to center.
     8. Warp the detected lane boundaries back onto the original image and output visual display of the lane boundaries and     numerical estimation of lane curvature and vehicle position.
     
## Required Libraries:
- Numpy
- Opencv
- Glob
- Matplotlib

## Step 1: Camera Calibration
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:
