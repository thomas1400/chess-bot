import chessbot.cvutil.transform as transform
import numpy as np
import cv2
import imutils
from .HoughBundler import HoughBundler


def find_largest_contour(bin_image):
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            if area > max_cnt_area:
                max_cnt_area = area
                best_cnt = cnt
    
    return best_cnt

def outline_contour(bin_image, contour):
    # outline largest contour, return
    return cv2.drawContours(np.zeros((bin_image.shape), np.uint8), [contour], 0, 255, 1)

def crop_to_chessboard(image, bin_thresh=180, border_size=3, output_size=600,debug=False):
    # grayscale, threshold to find white, then find the contour of the white background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY)
    bg_contour = find_largest_contour(thresh)
    if (bg_contour is None):
        return False, None

    # mask the image to the white background
    mask = np.ones((gray.shape), np.uint8) * 255
    cv2.drawContours(mask, [bg_contour], 0, 0, -1)
    cv2.drawContours(mask, [bg_contour], 0, 255, 3)
    masked = cv2.bitwise_or(gray, mask)

    # inv threshold to find black in the image, initialize board_outlined
    ret, board_outlined = cv2.threshold(masked, 100, 255, cv2.THRESH_BINARY_INV)

    # iteratively dilate, threshold dilation, find largest contour, find outline
    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    iterations = 4
    board_cnt = None
    for _ in range(iterations):
        dilated = cv2.dilate(board_outlined, kernel)
        ret, thresh = cv2.threshold(dilated, 10, 255, cv2.THRESH_BINARY)
        board_cnt = find_largest_contour(thresh)
        board_outlined = outline_contour(thresh, board_cnt)

    # fill the contour of the board, then iteratively erode to reach original board size pre-dilation
    board_filled = cv2.drawContours(np.zeros((gray.shape), np.uint8), [board_cnt], 0, 255, -1)
    for _ in range(iterations):
        board_filled = cv2.erode(board_filled, kernel)
    
    # find the contour of the filled board post-erosion
    board_cnt = find_largest_contour(board_filled)
    if (board_cnt is None):
        return False, None
    board_outlined = outline_contour(board_filled, board_cnt)

    # find lines on the board outline using HoughLinesP (the edges), then clean up using HoughBundler
    lines = cv2.HoughLinesP(board_outlined, rho=1.1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=100)
    hb = HoughBundler()
    merged_lines = hb.process_lines(lines, board_outlined)

    if debug:
        unmerged = image.copy()
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(unmerged, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(unmerged, (lines[i][0][0], lines[i][0][1]), 3, (255, 0, 0), -1)
            cv2.circle(unmerged, (lines[i][0][2], lines[i][0][3]), 3, (0, 255, 0), -1)
        merged = image.copy()
        for line in merged_lines:
            cv2.line(merged, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 3, cv2.LINE_AA)
        merged = imutils.resize(merged, width=400)
        unmerged = imutils.resize(unmerged, width=400)
        cv2.imshow("unmerged lines", unmerged)
        cv2.imshow("merged", merged)
        cv2.waitKey(0)

    # separate lines to horizontal and vertical, then find the four corners
    lines_x = []
    lines_y = []
    for line_i in merged_lines:
        try:
            line_i = line_i[0].tolist() + line_i[1].tolist()
        except:
            line_i = line_i[0] + line_i[1]
        
        orientation = hb.get_orientation(line_i)
        # if vertical
        if 45 < orientation < 135:
            lines_y.append(line_i)
        else:
            lines_x.append(line_i)

    corners = []
    for lx in lines_x:
        for ly in lines_y:
            x1, y1, x2, y2 = lx[0], lx[1], lx[2], lx[3]
            x3, y3, x4, y4 = ly[0], ly[1], ly[2], ly[3]
            ix = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            iy = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            corners.append([int(ix), int(iy)])
    
    if len(corners) != 4:
        return False, None
    
    # bring corners in by kernel size + border size to compensate for dilation and chessboard border
    compensation = kernel_size + border_size
    corners[0] = [corners[0][0] + compensation, corners[0][1] + compensation]
    corners[1] = [corners[1][0] + compensation, corners[1][1] - compensation]
    corners[2] = [corners[2][0] - compensation, corners[2][1] + compensation]
    corners[3] = [corners[3][0] - compensation, corners[3][1] - compensation]

    # warp image to fit chessboard
    ret, warped = transform.four_point_square_transform(image, corners, output_size)
    if not ret or warped is None:
        return False, None

    if debug:
        cv2.imshow("masked", masked)
        cv2.imshow("thresh", thresh)
        cv2.imshow("outlined", board_outlined)
        cv2.imshow("warped", warped)
        cv2.waitKey(0)
    
    return True, warped