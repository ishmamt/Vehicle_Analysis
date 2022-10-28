import detection as d
import tracker as t
import camera as c
import speed as s
import utilities as u
from tqdm import tqdm
import os
import cv2
import numpy as np
import math
import datetime


# Parameters to change
roi = [(85,1030), (1326, 1068), (768, 320), (727, 178), (808, 51), (745, 50), (727, 93), (535, 140), (427, 246)]
area = [(430, 297), (748, 304), (890, 520), (313, 537)]
deleting_line = [(256, 770), (1057, 750), (1194, 918), (246, 964)]
length = 25.0  # in meters
video_name = "recording.avi"
logger_name = "recording"
report_file_name = "recording"


# Flags to annotate or save the annotated video
show_video = False
save_video = True


detector = d.VehicleDetector(model_wights_path=os.path.join("yolov5", "models", "yolov5s.pt"))
tracker = t.VehicleTracker()
camera = c.Camera("Test", os.path.join("Data", video_name), roi)
frame_skipper = u.FrameSkipper(1)
logger = u.Logger(os.path.join("Data"), logger_name)
reporter = u.Reporter(os.path.join("Data", "Reports"), report_file_name, os.path.join("Data", "Frames"), logger, ['name', 'timestamp', 'speed(km/h)'])
speed = s.Speed(area, deleting_line, length, logger)
output_video = cv2.VideoWriter(os.path.join("Data", f"output_{video_name}"), 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                camera.fps, camera.size)


roi = camera.process_coordinates(roi)
area = camera.process_coordinates(area)
deleting_line = camera.process_coordinates(deleting_line)


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
            
            current_timestamp = math.floor(camera.video.get(cv2.CAP_PROP_POS_MSEC)/1000)
            current_timestamp = str(datetime.timedelta(seconds=int(current_timestamp)))

            masked_frame = speed.process_frame(masked_frame, tracked_objects_info, save_video, 
                                               FRAME_COUNT, camera.fps, current_timestamp, reporter)

        frame_skipper.increment_skipped_frame_count()
        frame_skipper.reset_skipped_frame_count()
        FRAME_COUNT += 1

        if save_video:
            cv2.polylines(masked_frame,[np.array([area[0], area[1], area[2], area[3]], np.int32)], True, (15, 220, 18), 6)
            cv2.polylines(masked_frame,[np.array([deleting_line[0], deleting_line[1], deleting_line[2], deleting_line[3]], np.int32)], True, (15, 220, 18), 6)
            
            if show_video:
                cv2.imshow("Video", masked_frame)
                k = cv2.waitKey(int(camera.fps * playback_speed))
                if k == ord('q'):
                    break
            
            output_video.write(masked_frame)

    elif p_bar.n == camera.total_frames:
        logger.info("Finished processing.")
        break

    else:
        frame_skipper.increment_skipped_frame_count()
        FRAME_COUNT += 1
        logger.error(f"Error at {FRAME_COUNT}. Frame could not be captured.")
        continue