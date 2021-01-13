import numpy as np
import imutils
import cv2


def mask_to_whitebg(image, threshold=120, filterStrength=150):

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # increase contrast with histogram equalization
    imageGray = cv2.equalizeHist(imageGray)

    # a filtering step to smooth noise in the image, then thresholding
    filtered = cv2.bilateralFilter(imageGray, 5, filterStrength, filterStrength)
    ret, thresh = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
    cv2.imshow("Contours", image)
    cv2.waitKey(0)

    contours = sorted(contours, key=lambda c: cv2.contourArea(c))

    return 