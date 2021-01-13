import numpy as np
import imutils
import cv2
from numpy.core.numeric import Infinity
from .HoughBundler import HoughBundler

def filter_clustered_lines(lines, number):
    # returns (retVal, lines)
    # retVal is False if there aren't enough lines to cluster

    if (len(lines) < number):
        # not enough lines were found to make a chessboard
        return (False, [])

    hb = HoughBundler()

    clusterStart = 0
    clusterEnd = number
    cluster = (clusterStart, clusterEnd)
    minClusterSpanSqr = Infinity
    while clusterEnd < len(lines):
        a, b = lines[clusterStart], lines[clusterEnd]
        ma, mb = [(a[0]+a[2])/2, (a[1]+a[3])/2], [(b[0]+b[2])/2, (b[1]+b[3])/2]
        distSqr = (mb[0]-ma[0])*(mb[0]-ma[0]) + (mb[1]-ma[1])*(mb[1]-ma[1])
        if distSqr < minClusterSpanSqr:
            cluster = (clusterStart, clusterEnd)
            minClusterSpanSqr = distSqr
        clusterStart += 1
        clusterEnd += 1

    return (True, lines[cluster[0]:cluster[1]])

def filter_chessboard_lines(lines):
    # find the 9 lines (both horizontally and vertically) that are most closely clustered
    # could find the median line, then find the 8 closest lines to the median
    # returns (retVal, hlines, vlines)

    hb = HoughBundler()

    lines_x = []
    lines_y = []
    for line_i in lines:
        line_i = line_i[0] + line_i[1]
        orientation = hb.get_orientation(line_i)
        # if vertical
        if 45 < orientation < 135:
            lines_y.append(line_i)
        else:
            lines_x.append(line_i)

    lines_y = sorted(lines_y, key=lambda line: line[1])
    lines_x = sorted(lines_x, key=lambda line: line[0])

    # horizontal lines
    ret_x, lines_x = filter_clustered_lines(lines_x, 9)
    ret_y, lines_y = filter_clustered_lines(lines_y, 9)

    if not ret_x or not ret_y:
        return (False, None, None)

    return (True, lines_x, lines_y)

def find_square_corners(image, lines_x, lines_y):
    # calculate square corners given horizontal and vertical lines

    corners = []
    for lx in lines_x:
        for ly in lines_y:
            x1, y1, x2, y2 = lx[0], lx[1], lx[2], lx[3]
            x3, y3, x4, y4 = ly[0], ly[1], ly[2], ly[3]
            ix = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            iy = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            corners.append([int(ix), int(iy)])
    
    for c in corners:
        cv2.circle(image, (c[0], c[1]), 3, (255, 0, 0), -1)

    cv2.imshow("Chessboard Corners", image)
    cv2.waitKey(0)

    print("[INFO] Done!")
    
    return corners

def find_chessboard(image, threshold=100, lineThreshold=100, filterStrength=100, minBoardLength=300, dilationSize=2, debug=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # increase contrast with histogram equalization
    imageGray = cv2.equalizeHist(imageGray)

    # a filtering step to smooth noise in the image, then thresholding
    filtered = cv2.bilateralFilter(imageGray, 5, filterStrength, filterStrength)
    ret, thresh = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY_INV)

    # find edges, dilate edges, then find lines
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    dilationKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilationSize+1, 2*dilationSize+1), (dilationSize, dilationSize))
    edges = cv2.dilate(edges, dilationKernel)
    lines = cv2.HoughLinesP(edges, rho=1.1, theta=np.pi/180, threshold=lineThreshold, minLineLength=minBoardLength, maxLineGap=100)

    if debug:
        # stack images for display
        rImageGray = imutils.resize(imageGray, width=350)  
        rFiltered = imutils.resize(filtered, width=350)  
        rThresh = imutils.resize(thresh, width=350)
        rEdges = imutils.resize(edges, width=350)  
        stacked = np.hstack([rImageGray, rFiltered, rThresh, rEdges])
        cv2.imshow("Chessboard Pre-processing", stacked)
        cv2.waitKey(0)

    # merge lines
    hb = HoughBundler()
    merged_lines = hb.process_lines(lines, image)
    retVal, lines_x, lines_y = filter_chessboard_lines(merged_lines)

    if debug:
        # display lines pre-merging
        unmerged = image.copy()
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(unmerged, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(unmerged, (lines[i][0][0], lines[i][0][1]), 3, (255, 0, 0), -1)
            cv2.circle(unmerged, (lines[i][0][2], lines[i][0][3]), 3, (0, 255, 0), -1)
        
        unmerged = imutils.resize(unmerged, width=400)

        # display lines without clustering
        merged = image.copy()
        for line in merged_lines:
            cv2.line(merged, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 3, cv2.LINE_AA)
        merged = imutils.resize(merged, width=400)

        stacked = np.hstack([unmerged, merged])
        cv2.imshow("HoughLinesP, HoughBundler", stacked)
        cv2.waitKey(0)
    
    if not retVal:
        print("[INFO] No chess board found")
        return False
    
    rImage = image.copy()
    for i in [lines_x, lines_y]:
        for line in i:
            cv2.line(rImage, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv2.LINE_AA)
        
    # display
    rImage = imutils.resize(rImage, width=400)
    cv2.imshow("Hough Line Detection", rImage)
    cv2.waitKey(0)

    find_square_corners(image, lines_x, lines_y)
    
    return True