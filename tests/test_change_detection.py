from . import context
context.init()

from chessbot.cvutil.chessboard_cv import crop_to_chessboard
import unittest
import cv2

class TestChessboardCV(unittest.TestCase):

    baseline_image = "images/neutralboard.jpg"
    move_images = [
        "images/blackonblack.jpg", "images/whiteonblack.jpg", "images/blackonwhite.jpg", 
        "images/whiteonblack.jpg"
        ]

    def test_image_read(self):
        baseline = cv2.imread(self.baseline_image)
        self.assertIsNotNone(baseline)

        for path in self.move_images:
            image = cv2.imread(path)
            self.assertIsNotNone(image)
    
    def test_chessboard_detection(self):
        for path in self.move_images:
            try:
                image = cv2.imread(path)
                ret, cropped = crop_to_chessboard(image)
                self.assertTrue(ret)
            except:
                self.fail()

if __name__ == "__main__":
    unittest.main()