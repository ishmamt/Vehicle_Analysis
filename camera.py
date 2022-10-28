import cv2
import secrets
import os
import numpy as np


class Camera():
    '''
    Parent class for all cameras.
    '''
    
    def __init__(self, name, video_path, roi, processed_until=None):
        '''
        Constructor for Camera class.

            Parameters:
                name (string): Name of the camera.
                video_path (string): Path to the video file.
                status (boolean): Whether the camera is active or not.
                roi (list): Co-ordinates for region of interest.
                processed_until (int): A checkpoint denoting number of frames already processed.
        '''
        self.id = secrets.token_hex(8)
        self.name = name
        self.video_path = video_path
        self.roi = roi
        self.processed_until = processed_until

        if not os.path.exists(self.video_path):
            raise FileNotFoundError("Invalid path to video.")

        self.video = cv2.VideoCapture(self.video_path)
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.video.get(3)), int(self.video.get(4)))

    
    def get_masked_frame(self, frame):
        '''
        Returns a masked frame based on the roi.

            Parameters:
                frame (numpy array): A frame from the video.

            Returns:
                masked_frame (numpy array): Masked frame only showing the roi.
        '''
        mask = np.zeros(frame.shape, dtype=np.uint8)
        roi_corners = np.array([self.roi], dtype=np.int32)
        
        # Fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your frame
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)

        # Applying the mask
        masked_frame = cv2.bitwise_and(frame, mask)
        
        return masked_frame
    
    
    def process_coordinates(self, area, display_dimension=(1920, 1080)):
        '''
        Scales the frame area co-ordinates according to the video dimensions and display dimensions.
        
            Parameters:
                area (list): Co-ordinates to be scaled.
                display_dimension (tuple): Dimension of the display.
            
            Returns:
                area_processed (list): List of scaled co-ordinates.
        '''
        x_factor = self.size[0] / display_dimension[0]
        y_factor = self.size[1] / display_dimension[1]
        area_processed = []
        for co in area:
            area_processed.append((int(co[0] * x_factor), int(co[1] * y_factor)))
        
        return area_processed