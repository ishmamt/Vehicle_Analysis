import detection as d
import cv2

img = cv2.imread("Data/test.png")
detector = d.VehicleDetector()

result = detector.detect(img)
cv2.imwrite("asda.png" ,result)
