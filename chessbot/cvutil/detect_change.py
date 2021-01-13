from .align import align_images
import numpy as np
import imutils
import cv2

def detect_change(image, baseline, minWidth=10, minHeight=10):

    image = align_images(image, baseline)

    baselineGray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image before histogram equalization", imageGray)

    baselineGray = cv2.equalizeHist(baselineGray)
    imageGray = cv2.equalizeHist(imageGray)
    # cv2.imshow("image after histogram equalization", imageGray)
    # cv2.waitKey(0)

    diff = cv2.absdiff(imageGray, baselineGray)
    cv2.imshow("diff", diff)
    cv2.waitKey(0)

    ret, thresh = cv2.threshold(diff, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    threshContours = []
    # eliminate contours below size threshold
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        width = rect[1][0]
        height = rect[1][1]

        if (width >= minWidth and height >= minHeight):
            threshContours.append(cnt)
            print("w: " + str(width) + " h: " + str(height))

    contours = threshContours

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contours on Difference", image)
    cv2.waitKey(0)

    return contours