import detection as d
import tracker as t
import cv2

detector = d.VehicleDetector()
tracker = t.VehicleTracker()

img = cv2.imread("test2.jpg")

# for _ in range(0, 100):
#     results = detector.get_detection_results(img)
#     print(tracker.track(results, img))
#     print("\n")

cv2.imwrite("result2.jpg", detector.detect(img))