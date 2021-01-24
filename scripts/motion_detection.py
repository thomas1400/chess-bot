import context
context.init()

import time
import numpy as np
import cv2


def capture_frame():
    print("motionless!")

def main():

    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    motionless = False
    motionless_start = None

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

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
                    capture_frame()
                    motionless = True
                    motionless_start = None
        else:
            motionless = False
            motionless_start = None
        
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

        prev = gray

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()