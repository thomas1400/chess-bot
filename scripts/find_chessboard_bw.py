import context
context.init()

from chessbot.cvutil.find_chessboard import find_bw_chessboard
from chessbot.cvutil.transform import four_point_square_transform
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image for chessboard")
args = vars(ap.parse_args())

print("[INFO] loading image...")
image = cv2.imread(args["image"])
if image is None:
    print("[ERROR] Image path is invalid")
    exit(1)

print("[INFO] identifying chessboard...")

# mask_to_whitebg(image)
retVal, corners = find_bw_chessboard(image.copy(), threshold=80, lineThreshold=150, filterStrength=150, debug=True)

# find the four outer corners, warp to that
warped = four_point_square_transform(image.copy(), [corners[0], corners[8], corners[72], corners[80]], 400)
cv2.imshow("Warped", warped)
cv2.waitKey(0)