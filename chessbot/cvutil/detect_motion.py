import numpy as np
import cv2
import time

def wait_til_motionless(capture):

    ret, frame = capture.read()
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    motionless = False
    motionless_start = None

    while(True):
        # Capture frame-by-frame
        ret, frame = capture.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev)
        ret, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            if not motionless:
                # require half a second of no motion to say frame is "motionless"
                if motionless_start is None:
                    motionless_start = time.process_time()
                if time.process_time() - motionless_start >= 1:
                    return

        else:
            motionless = False
            motionless_start = None
