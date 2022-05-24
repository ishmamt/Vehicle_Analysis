from unittest import result
import detection as d
import tracker as t
import cv2

detector = d.VehicleDetector()
tracker = t.VehicleTracker()

img = cv2.imread("test.jpg")

for _ in range(0, 6):
    results = detector.get_detection_results(img)
    print(tracker.track(results, img))
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")