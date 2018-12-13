import cv2
import numpy as np
import imutils
from customizedTrackBar import customizedTrackBar

neighborSizeBar = customizedTrackBar(3, 2, 95, 75, "Neighborhood Size", "Tuning")
tuneThreshBar = customizedTrackBar(0, 1, 20, 10, "Tune Threshold Value", "Tuning")
circleRadiusBar = customizedTrackBar(0, 1, 200, 3, "Approximate Radius", "Tuning")
minDistBar = customizedTrackBar(1, 1, 500, 100, "Minimum Distance", "Tuning")

image = cv2.imread("eye-pupil2.jpg")
image = imutils.resize(image, height=400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred1 = cv2.GaussianBlur(gray, (51,51), 0)
blurred2 = cv2.medianBlur(gray, 51)

while True:
    tuneThresh = tuneThreshBar.getValue()
    neighborSize = neighborSizeBar.getValue()

    radius = circleRadiusBar.getValue()
    minDistance = minDistBar.getValue()

    threshImage = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, neighborSize, tuneThresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(threshImage, kernel, 3)

    edgeMap = imutils.auto_canny(threshImage, sigma=0.33)
    detectedCircles = cv2.HoughCircles(blurred2, cv2.HOUGH_GRADIENT, 1.2, minDistance)

    grayClone = gray.copy() 
    if detectedCircles is not None:
        detectedCircles = np.round(detectedCircles[0, :]).astype("int")

        for x, y, r in detectedCircles:
            cv2.circle(grayClone, (x, y), r, (255,0,0), 4)
            cv2.circle(edgeMap, (x, y), r, (255,0,0), 4)

    else:
        print("There was no circles detected in this image")

    comparison1 = np.hstack((grayClone, edgeMap))
    comparison2 = np.hstack((threshImage, dilated))
    cv2.imshow("Detecting Circles Using Hough Transform", comparison1)
    cv2.imshow("Detecting Circles Using Hough Transform2", comparison2)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break