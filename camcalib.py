import numpy as np
import cv2
import glob


# prepare object points
nx = 9  # the number of inside corners in x
ny = 6  # the number of inside corners in y

# Read in and make list of calibration images
images = glob.glob('camera_cal\calibration*.jpg')


def calibrateCamera(glob):

    # arrays to store object point and image points
    objpoints = []
    imgpoints = []

    # prepare object points

    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for frame in images:
        img = cv2.imread(frame)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[1::-1]
        # cv2.imshow('gray',gray)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if(ret):
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # cv2.imshow('corners',img)
            cv2.waitKey(25)

    return imgpoints, objpoints, shape


imgpoints, objpoints, shape = calibrateCamera(images)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
print "mtx:", mtx
print "dist:", dist

fname = 'camera_cal\calibration2.jpg'
img = cv2.imread(fname)
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imshow('original', img)
cv2.imshow('calibrated', dst)
cv2.waitKey(0)
