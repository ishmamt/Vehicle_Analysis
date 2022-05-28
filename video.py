import cv2
import secrets


class Camera():
    ''' Parent class for all cameras.
    
    '''
    
    def __init__(self, name, status=True, resolution=(0, 0), roi=[], processed_until=None):
        ''' Initialize the camera
        Parameters:
            name: string; Name of the camera.
            status: boolean; Whether the camera is active or not.
            resolution: tuple; Resolution of the camera.
            roi: list; Co-ordinates for region of interest.
            processed_until: int; A checkpoint denoting number of frames already processed.
        '''
        
        self.id = secrets.token_hex(8)
        self.name = name
        self.status = status
        self.resolution = resolution
        self.roi = roi
        self.processed_until = processed_until
    