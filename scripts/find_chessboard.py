import context
context.init()

from chessbot.cvutil.chessboard_cv import crop_to_chessboard
from chessbot.cvutil.transform import four_point_transform
from chessbot.cvutil.detect_change import detect_change
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--move", required=True, help="path to image of move")
ap.add_argument("-p", "--previous", required=True, help="path to image of previous board state")
args = vars(ap.parse_args())

print("[INFO] loading move image...")
move = cv2.imread(args["move"])
prev = cv2.imread(args["previous"])
if move is None or prev is None:
    print("[ERROR] Image path is invalid")
    exit(1)

print("[INFO] identifying chessboards...")

# crop images to chessboard
retMove, cropped_move = crop_to_chessboard(move, debug=False)
retPrev, cropped_prev = crop_to_chessboard(prev, debug=False)

# align then do change detection to find moves
if retMove and retPrev:
    # aligned_move = align.align_images(cropped_move, cropped_prev, keepPercent=0.2, debug=False)
    overlay = cropped_prev.copy()
    output = cropped_move.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    cv2.imshow("overlaid and aligned", output)
    cv2.waitKey(0)

    detect_change(cropped_move, cropped_prev)

else:
    print("[ERROR] chessboard not found in one or more image(s)")
    exit(1)