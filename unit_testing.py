import detection as d
import tracker as t
import camera as c
import speed as s
import utilities as u
from tqdm import tqdm
import os
import cv2
import numpy as np

roi = [(85,1030), (1326, 1068), (768, 320), (727, 178), (808, 51), (745, 50), (727, 93), (535, 140), (427, 246)]
entry_area = [(423, 288), (753, 297), (768, 334), (414, 321)]
exit_area = [(320, 535), (880, 530), (890, 560), (315, 562)]
deleting_line = [(256, 770), (1057, 750), (1194, 918), (246, 964)]
length = 25.0  # in meters




detector = d.VehicleDetector()
tracker = t.VehicleTracker()
camera = c.Camera("Test", os.path.join("Data", "test_10s.avi"), roi)
frame_skipper = u.FrameSkipper(1)
logger = u.Logger(os.path.join("Data"), "test_10seconds")
speed = s.Speed(entry_area, exit_area, deleting_line, length, logger)
output_video = cv2.VideoWriter(os.path.join("Data", "output_10s.avi"), 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                camera.fps, camera.size)

x_numpy = camera.size[0]
y_numpy = camera.size[1]

x_editor = 1920
y_editor = 1080

x_factor = x_numpy/x_editor
y_factor = y_numpy/y_editor

def process_coordinates(area):
    area_processed = []
    for co in area:
        area_processed.append((int(co[0]*x_factor), int(co[1]*y_factor)))
    return area_processed

roi = process_coordinates(roi)
entry_area = process_coordinates(entry_area)
exit_area = process_coordinates(exit_area)
deleting_line = process_coordinates(deleting_line)

FRAME_COUNT = 0
p_bar = tqdm(total = camera.total_frames)
logger.info("Starting processing frames.")
annotate = True
playback_speed = 1

while True:
    success, frame = camera.video.read()

    if success:
        p_bar.update(1)

        if frame_skipper.if_process_frame():
            masked_frame = camera.get_masked_frame(frame)
            results = detector.get_detection_results(masked_frame)
            tracked_objects_info = tracker.track(results, masked_frame, confidence_threshold=0.5)

            masked_frame = speed.process_frame(masked_frame, tracked_objects_info, True, FRAME_COUNT, camera.fps)



        frame_skipper.increment_skipped_frame_count()
        frame_skipper.reset_skipped_frame_count()
        FRAME_COUNT += 1

        cv2.polylines(masked_frame,[np.array([entry_area[0], entry_area[1], entry_area[2], entry_area[3]], np.int32)], True, (15, 220, 18), 6)
        cv2.polylines(masked_frame,[np.array([exit_area[0], exit_area[1], exit_area[2], exit_area[3]], np.int32)], True, (15, 220, 18), 6)
        # cv2.line(masked_frame, deleting_line[0], deleting_line[1], (15, 220, 18), 6)
        cv2.polylines(masked_frame,[np.array([deleting_line[0], deleting_line[1], deleting_line[2], deleting_line[3]], np.int32)], True, (15, 220, 18), 6)
        #cv2.imshow("Video", masked_frame)
        output_video.write(masked_frame)

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