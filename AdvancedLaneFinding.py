
import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')


def undistort(image):
    mtx = np.array([[1.15777818e+03, 0.00000000e+00, 6.67113857e+02],
                    [0.00000000e+00, 1.15282217e+03, 3.86124583e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist = np.array(
        [[-0.24688507, -0.02373156, -0.00109831, 0.00035107, -0.00259866]])

    dst_img = cv2.undistort(image, mtx, dist, None, mtx)

    return dst_img


def blur(image, kernel):
    blur = cv2.GaussianBlur(image, kernel, 0)
    return blur


def drawROI(image, pts):
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True, (0, 255, 255))


def selectROI(image, pts):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [pts], (255,) * 3)
    return np.bitwise_and(mask, image)


def warp(image):
    img_size = (int(image.shape[1]), int(image.shape[0]))
    # print img_size
    src = np.array([[190, 700], [580, 450], [
                   730, 450], [1160, 700]], np.float32)
    dst = np.array([[380, 720], [380, 10], [950, 10], [950, 720]], np.float32)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, img_size)

    # Return the resulting image and matrix
    return warped

def unwarp(image, img_size):
    # img_size = (int(image.shape[1]), int(image.shape[0]))
    # print img_size
    src = np.array([[190, 700], [580, 450], [
                   730, 450], [1160, 700]], np.float32)
    dst = np.array([[380, 720], [380, 10], [950, 10], [950, 720]], np.float32)
    # Given src and dst points, calculate the perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, Minv, img_size)

    # Return the resulting image and matrix
    return warped


def cvtGray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def showImage(image, winName):
    cv2.imshow(winName, image)


def cvtHLS(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def cvtLAB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def selectChannel(image, channel):
    if(channel == 0):
        return image[:, :, 0]
    elif(channel == 1):
        return image[:, :, 1]
    elif(channel == 2):
        return image[:, :, 2]


def applyBinaryThresh(image, lower, upper, inverted=False):
    if (inverted):
        binary_output = np.ones_like(image)
        binary_output[((image > lower) & (image <= upper))] = 0
    else:
        binary_output = np.zeros_like(image)
        binary_output[((image > lower) & (image <= upper))] = 1

    return binary_output


def combineThresh(img1, img2):
    image = np.bitwise_or(img1, img2)
    return image


def absSobel(image, lower, upper, orient='x'):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sobel_binary = applyBinaryThresh(scaled_sobel, lower, upper)
    return sobel_binary


def preProcess(image):
    hls_image = cvtHLS(image)
    lab_image = cvtLAB(image)
    b_binary = (applyBinaryThresh(selectChannel(lab_image, 2), 111, 220, True))
    # l_binary = (applyBinaryThresh(selectChannel(hls_image, 1), 100, 240))
    s_binary = (applyBinaryThresh(selectChannel(hls_image, 2), 175, 255))

    color = combineThresh(b_binary, s_binary) * 255
    sobel_x = absSobel(image, 30, 110, 'x') * 255
    sobel_y = absSobel(image, 30, 110, 'y') * 255
    sobel_abs = np.bitwise_and(sobel_x, sobel_y)
    combined = combineThresh(color, sobel_abs)
    return combined


def sliding_window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low,
                               win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = [rectangle_data, histogram, binary_warped]

    return leftx, rightx, lefty, righty, left_fit, right_fit, visualization_data


def showSWsearch(left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data):
    ploty = np.linspace(
        0, visualization_data[2].shape[0] - 1, visualization_data[2].shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    out_img = np.dstack(
        (visualization_data[2], visualization_data[2], visualization_data[2])) * 255
    nonzero = visualization_data[2].nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    leftp = zip(left_fitx, ploty)
    leftp = np.array([leftp], dtype=np.int32)

    rightp = zip(right_fitx, ploty)
    rightp = np.array([rightp], dtype=np.int32)

    rectangles = visualization_data[0]
    for rect in rectangles:
        cv2.rectangle(out_img, (rect[2], rect[0]), (rect[3], rect[1]),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (rect[4], rect[0]), (rect[5], rect[1]),
                      (0, 255, 0), 2)

    cv2.polylines(out_img, leftp, False, color=[0, 255, 0],
                  thickness=3, lineType=8, shift=0)
    cv2.polylines(out_img, rightp, False, color=[0, 255, 0],
                  thickness=3, lineType=8, shift=0)

    cv2.imshow("out", out_img)


def swNeighbourSearch(binary_warped, left_fit, right_fit):
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
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
    visualization_data = [binary_warped, margin]
    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    return leftx, rightx, lefty, righty, left_fit, right_fit, visualization_data


def showNeighbourSearch(leftx, rightx, lefty, righty, left_fit, right_fit, visualization_data):
    # Create an image to draw on and an image to show the selection window
    binary_warped = visualization_data[0]
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    margin = visualization_data[1]
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    cv2.imshow("result", result)


def findCurvature(binary_warped, leftx, rightx, lefty, righty):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 0.04285   # meters per pixel in y dimension
    xm_per_pix = 0.005285  # meters per pixel in x dimension
    y_eval = np.max(ploty)
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    # print(left_curverad, right_curverad)
    if left_fit_cr is not None and right_fit_cr is not None:
        car_position = binary_warped.shape[1] / 2
        h = binary_warped.shape[0]
        l_fit_x_int = left_fit_cr[0]*h**2 + left_fit_cr[1]*h + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*h**2 + right_fit_cr[1]*h + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix

    return left_curverad, right_curverad, center_dist


def drawLane(warped, left_fit, right_fit, image_shape, undist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp, image_shape)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # cv2.imshow("lane", result)
    return result


def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    # h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40, 70), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40, 120), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)
    return new_img


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def addFit(self, fit):
        self.current_fit = fit
        if self.current_fit is not None:
            # print self.best_fit
            if self.best_fit is not None:
                # Compare with Best Fit
                self. diffs = abs(self.current_fit - self.best_fit)
            if((self.diffs[0] > 0.001 - self.diffs[1] > 1 - self.diffs[2] > 100).any()):
                # discard current
                self.detected = False
            else:
                self.detected = True
                self.recent_xfitted.append(self.current_fit)
                if len(self.recent_xfitted) > 8:
                    self.recent_xfitted = self.recent_xfitted[len(self.recent_xfitted) - 8:]

                self.best_fit = np.average(self.recent_xfitted, axis=0)
                self.bestx.append(self.best_fit)
                # print self.best_fit
        else:
            self.detected = False
            if len(self.recent_xfitted) > 0:
                self.recent_xfitted = self.recent_xfitted[:len(self.recent_xfitted) - 1]
            self.best_fit = np.average(self.recent_xfitted, axis=0)
            self.bestx.append(self.best_fit)





def findLanes(image):
    img_size = (int(image.shape[1]), int(image.shape[0]))
    # cv2.imshow("orginal", image)
    image = undistort(image)
    undist = image
    # cv2.imshow("undistorted", image)
    # cv2.imshow("blurred", image)
    roi_pts = np.array([[190, 700], [580, 450], [730, 450], [1160, 700]], np.int32)
    # drawROI(image, roi_pts)
    # cv2.imshow("ROI", image)
    image = preProcess(image)
    # cv2.imshow("preProcessed", image)
    image = selectROI(image, roi_pts)
    warped = warp(image)
    # cv2.imshow("Warped", warped)
    if not Left.detected or not Right.detected:
        leftx, rightx, lefty, righty, left_fit, right_fit, visualization_data = sliding_window_search(warped)
    else:
        leftx, rightx, lefty, righty, left_fit, right_fit, visualization_data = swNeighbourSearch(warped, Left.best_fit, Right.best_fit)

    if left_fit is not None and right_fit is not None:
        # Calculate the intercepts
        h = warped.shape[0]
        l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        int_diff = abs(l_fit_x_int - r_fit_x_int)
        # print int_diff
        diff = abs(int_diff - 500)
        print diff

        if not ((diff) > 50):
            # Do Something
            Left.addFit(left_fit)
            Right.addFit(right_fit)
        else:
            Left.best_fit = Left.bestx[-3]
            Right.best_fit = Right.bestx[-3]



    left_curverad, right_curverad, center_dist = findCurvature(warped, leftx, rightx, lefty, righty)
    result = drawLane(warped, Left.best_fit, Right.best_fit, img_size, undist)
    result = draw_data(result, left_curverad, center_dist)
    # print Right.current_fit
    # print Right.detected
    # print Right.recent_xfitted

    return result


Right = Line()
Left = Line()
cap = cv2.VideoCapture('project_video.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    result = findLanes(frame)
    cv2.imshow("result", result)

    if cv2.waitKey(8) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# image = cv2.imread("test_images/test5.jpg")

# cv2.waitKey(0)
