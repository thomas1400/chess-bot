import context
context.init()

from chessbot.cvutil.align import align_images
from chessbot.cvutil.detect_change import detect_change
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True, help="path to template image")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
if image is None or template is None:
    print("[ERROR] image or template path is invalid")
    exit(1)

print("[INFO] aligning images...")
aligned = align_images(image, template, debug=False)

# resize images
aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)

# side-by-side
stacked = np.hstack([aligned, template])

# overlaid
overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

# show the two visualizations
cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlaid", output)
cv2.waitKey(0)

contours = detect_change(image, template)