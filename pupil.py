import cv2
import numpy as np
# custom class that I have made using TrackBar in OpenCV
from customizedTrackBar import customizedTrackBar 
"""
When you first initialize an instance of customizedTrackBar class which I made,
    you need to pass the following parameters:
    (startValue, step, endValue, defaultValue, variableName, windowName)

    Examples are: following param1Bar & param2Bar
"""
param1Bar = customizedTrackBar(20, 1, 255, 45, "param1", "Tuning") # Initialize a bar to tune param1 value
param2Bar = customizedTrackBar(20, 1, 50, 35, "param2", "Tuning") # Initialize a bar to tune param2 value

image = cv2.imread("eye-pupil.jpg") # load an image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert image into gray scale
blurred = cv2.GaussianBlur(gray, (51,51), 0) # apply filter on the image with a kernel size 51x51

while True: # keep looping for the user to be able to tune the parameters
    imageClone = image.copy() # get a copy of the image
    maxThresh = param1Bar.getValue() # get maxThresh value of trackBar to be used in HoughCircles()
    param2 = param2Bar.getValue() # get param2 value of trackBar to be used in HoughCircles()
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=maxThresh, param2=param2, minRadius=0, maxRadius=0)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        for x, y, r in circles:
            # draw the circle in blue color in the output image, then draw a rectangle
            cv2.circle(imageClone, (x, y), r, (255,0,0), 2)
            # draw a very small circle in red color representing the circle center
            cv2.circle(imageClone, (x, y), 1, (0,0,255), 2)

    else:
        print("There was no circles detected in this image")

    cv2.imshow("Detecting Circles Using Hough Transform", imageClone)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): # break the loop if the user presses on "q" button
        break