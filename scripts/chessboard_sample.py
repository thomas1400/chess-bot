import context
context.init()

from chessbot.cvutil.find_chessboard import find_chessboard
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
find_chessboard(image, threshold=80, lineThreshold=150, filterStrength=150, debug=True)