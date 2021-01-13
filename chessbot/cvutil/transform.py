import numpy as np
import imutils
import cv2


def order_points(pts):
    # sort top to bottom, left to right
    ordered = sorted(pts, key=lambda pt: (pt[1], pt[0]))
    return np.float32(ordered)

def four_point_transform(image, pts, size):
    rect = order_points(pts)

    # get destination points from size, sort to ensure consistent order
    dst = order_points([[0, 0],[size - 1, 0],[size - 1, size - 1],[0, size - 1]])

    # get the perspective transform matrix and then apply it
    T = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, T, (size, size))
    # return the warped image
    return warped
