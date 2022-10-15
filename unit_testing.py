import detection as d
import tracker as t
import camera as c
import utilities as u
from tqdm import tqdm
import os

detector = d.VehicleDetector()
tracker = t.VehicleTracker()
camera = c.Camera("Test", os.path.join("Data", "recording.avi"), [(0,325), (0,772), (1100,260), (850,200)])
frame_skipper = u.FrameSkipper(4)
logger = u.Logger(os.path.join("Data"), "recording")

FRAME_COUNT = 0
p_bar = tqdm(total = camera.total_frames)
logger.info("Starting processing frames.")

while True:
    success, frame = camera.video.read()

    if success:
        p_bar.update(1)

        if frame_skipper.if_process_frame():
            frame = camera.get_masked_frame(frame)

        frame_skipper.increment_skipped_frame_count()
        frame_skipper.reset_skipped_frame_count()
        FRAME_COUNT += 1

    elif p_bar.n == camera.total_frames:
        logger.info("Finished processing.")
        break

    else:
        frame_skipper.increment_skipped_frame_count()
        FRAME_COUNT += 1
        logger.error(f"Error at {FRAME_COUNT}.")
        continue