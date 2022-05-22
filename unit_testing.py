import detection as d
import cv2

detector = d.Detector()
detector.set_detection_model()

img = cv2.imread("test.jpg")

img = detector.detect(img)

cv2.imwrite("test_result.jpg", img)