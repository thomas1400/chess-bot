import numpy as np
import cv2

def gray_to_bwg(gray, w_thresh=225, b_thresh=30):
    # converts a grayscale image to black-white-gray
    ret, w_img = cv2.threshold(gray, w_thresh, 255, cv2.THRESH_BINARY)
    ret, b_img = cv2.threshold(gray, b_thresh, 255, cv2.THRESH_BINARY_INV)
    b_img = cv2.bitwise_not(b_img)
    bwg = np.ones((gray.shape), np.uint8) * 122
    bwg = cv2.bitwise_and(bwg, b_img)
    bwg = cv2.bitwise_or(bwg, w_img)
    return bwg

def detect_change(change, baseline, minArea=400, minWidth=15, minHeight=15):

    base_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    chng_gray = cv2.cvtColor(change, cv2.COLOR_BGR2GRAY)

    base_gray = cv2.GaussianBlur(base_gray, (9, 9), 0)
    chng_gray = cv2.GaussianBlur(chng_gray, (9, 9), 0)

    # convert grayscale to black-white-gray images (ternary)
    base_bwg = gray_to_bwg(base_gray)
    chng_bwg = gray_to_bwg(chng_gray)

    diff = cv2.absdiff(base_bwg, chng_bwg)
    diff = cv2.erode(diff, np.ones((5, 5), np.uint8))
    # cv2.imshow("diff", diff)
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    thresh_contours = []
    # eliminate contours below size threshold
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area = cv2.contourArea(cnt)
        width = rect[1][0]
        height = rect[1][1]

        if (area > minArea and width > minWidth and height > minHeight):
            thresh_contours.append(cnt)

    contours = thresh_contours

    cv2.drawContours(change, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contours on Difference", change)
    cv2.waitKey(0)

    return contours