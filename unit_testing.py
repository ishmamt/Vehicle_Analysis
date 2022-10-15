import detection as d
import tracker as t
import camera as c
import speed as s
import utilities as u
from tqdm import tqdm
import os
import cv2
import numpy as np

roi = [(470, 165), (715, 180), (1230, 990), (200, 1030)]
entry_area = [(450, 300), (675, 300), (700, 350), (420, 350)]
exit_area = [(310, 530), (740, 530), (850, 740), (270, 740)]
deleting_line = [(210, 780), (1145, 800), (1145, 880), (210, 880)]
distacne_line = [(1077,384), (1125,394), (835, 966), (718,943)]
length = 25.0  # in meters

detector = d.VehicleDetector()
tracker = t.VehicleTracker()
camera = c.Camera("Test", os.path.join("Data", "recording.avi"), roi)
frame_skipper = u.FrameSkipper(4)
logger = u.Logger(os.path.join("Data"), "recording")
speed = s.Speed(entry_area, exit_area, deleting_line, distacne_line, length, logger)

FRAME_COUNT = 0
p_bar = tqdm(total = camera.total_frames)
logger.info("Starting processing frames.")
annotate = True
playback_speed = 0.1

while True:
    success, frame = camera.video.read()

    if success:
        p_bar.update(1)

        if frame_skipper.if_process_frame():
            masked_frame = camera.get_masked_frame(frame)
            results = detector.get_detection_results(masked_frame)
            tracked_objects_info = tracker.track(results, masked_frame)

            masked_frame = speed.process_frame(masked_frame, tracked_objects_info, True, FRAME_COUNT, camera.fps)



        frame_skipper.increment_skipped_frame_count()
        frame_skipper.reset_skipped_frame_count()
        FRAME_COUNT += 1

        cv2.polylines(masked_frame,[np.array([entry_area[0], entry_area[1], entry_area[2], entry_area[3]], np.int32)], True, (15, 220, 18), 6)
        cv2.polylines(masked_frame,[np.array([exit_area[0], exit_area[1], exit_area[2], exit_area[3]], np.int32)], True, (15, 220, 18), 6)
        # cv2.line(masked_frame, deleting_line[0], deleting_line[1], (15, 220, 18), 6)
        cv2.polylines(masked_frame,[np.array([deleting_line[0], deleting_line[1], deleting_line[2], deleting_line[3]], np.int32)], True, (15, 220, 18), 6)
        cv2.imshow("Video", masked_frame)

        k = cv2.waitKey(int(camera.fps * playback_speed))
        if k == ord('q'):
            break

    elif p_bar.n == camera.total_frames:
        logger.info("Finished processing.")
        break

    else:
        frame_skipper.increment_skipped_frame_count()
        FRAME_COUNT += 1
        logger.error(f"Error at {FRAME_COUNT}.")
        continue