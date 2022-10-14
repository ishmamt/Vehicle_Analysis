import cv2
import secrets
import os


class Camera():
    '''
    Parent class for all cameras.
    '''
    
    def __init__(self, name, video_path, roi=None, processed_until=None):
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
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
