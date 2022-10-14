import os
import torch
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.utils.general import xyxy2xywh


class Tracker():
    '''
    Parent class for object tracking.
    
        Attributes:
            model (model object): Tracker model.
    '''

    def __init__(self):
        '''
        Constructor for Tracker class.
        '''
        self.model = None

    
    def set_tracker_model(self, model_name, config_path):
        '''
        Sets the model for tracking.
        
            Parameters:
                model_name (string): Name of the model.
                config_path (string): Path to the config.yaml file.
        '''
        if not os.path.exists(config_path):
            raise FileNotFoundError("Invalid path to config file.")

        config = get_config()
        config.merge_from_file(config_path)

        self.model = DeepSort(model_name, max_dist=config.DEEPSORT.MAX_DIST, max_iou_distance=config.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=config.DEEPSORT.MAX_AGE, n_init=config.DEEPSORT.N_INIT, nn_budget=config.DEEPSORT.NN_BUDGET, use_cuda=True)

    
    def get_tracker_model(self):
        '''
        Fetches the tracker model being used.
        
        Returns:
            model (model object): The model attribute of the tracker class.
        ''' 

        return self.model

    
    def get_tracker_ids(self, results, image, confidence_threshold):
        '''
        Fetches the tracker ids using deepsort model.
        
            Parameters:
                results (pytorch tensor object): The results of inference on the given image.
                image (numpy array): Image to run inference on.
                confidence_threshold (float): Threshold for minimum confidence of detection.

            Returns:
                tracker_ids (list): List of tracked objects.
        '''
        results = results.xyxy[0].clone()  # Copy of results tensor, to avoid changing the original tensor.
        filtered_results = results[torch.where(results[:, 4] > confidence_threshold)]  # Filtered to only include above a certain threshold.
        xywhs = xyxy2xywh(filtered_results[:, 0:4])
        confidences = filtered_results[:, 4]
        classes =  filtered_results[:, 5]
        
        return self.model.update(xywhs.cpu(), confidences.cpu(), classes.cpu(), image)
    

    def get_tracked_objects_info(self, tracked_objects):
        '''
        Returns the bounding box locations and tracker ids for tracked objects.
        
            Parameters:
                tracked_objects (list): List of tracked objects.

            Returns:
                tracked_objects_info (list): List of tuples containing info about tracked objects.
        '''
        tracked_objects_info = []

        for i in range(len(tracked_objects)):
            xmin = tracked_objects[i][0] 
            ymin = tracked_objects[i][1] + 10 
            xmax = tracked_objects[i][2] 
            ymax = tracked_objects[i][3] 
            
            id = int(tracked_objects[i][4])
            
            tracked_objects_info.append((xmin, ymin, xmax, ymax, id))
        
        return tracked_objects_info

    
    def track(self, results, image, confidence_threshold = 0.5):
        '''
        Returns a list containing bounding boxes and track ids of tracked objects.
        
            Parameters:
                results (pytorch tensor object): The results of inference on the given image.
                image (numpy array): Image to run inference on.
                confidence_threshold (float): Threshold for minimum confidence of detection.
            
            Returns:
                tracked_objects_info (list): List of tuples containing info about tracked objects.
        '''
        tracked_objects = self.get_tracker_ids(results, image, confidence_threshold)

        return self.get_tracked_objects_info(tracked_objects)


class VehicleTracker(Tracker):
    '''
    Vehicle tracker class. Inherits from Tracker class.
    '''

    def __init__(self, model_name="osnet_x0_25", config_path=os.path.join("deep_sort", "configs", "deep_sort.yaml")):
        '''
        Vehicle tracker class. Inherits from Tracker class.
        
            Parameters:
                model_name (string): Name of the model.
                config_path (string): Path to the config file.
        '''
        super().__init__()
        self.set_tracker_model(model_name, config_path)

    
    def get_tracker_ids(self, results, image, confidence_threshold):
        '''
        Fetches the tracker ids for vehicles using deepsort model.

            Parameters:
                results (pytorch tensor object): The results of inference on the given image.
                image (numpy array): Image to run inference on.
                confidence_threshold (float): Threshold for minimum confidence of detection.
                
            Returns:
                model (model object): The model attribute of the tracker class.
        '''
        results = results.xyxy[0].clone()  # Copy of results tensor, to avoid changing the original tensor.
        filtered_results = results[torch.where((results[:, 5] > 0) & (results[:, 5] < 8) & (results[:, 4] > confidence_threshold))]  # Filtered to only include vehicles and above a certain threshold.
        xywhs = xyxy2xywh(filtered_results[:, 0:4])
        confidences = filtered_results[:, 4]
        classes =  filtered_results[:, 5]
        
        return self.model.update(xywhs.cpu(), confidences.cpu(), classes.cpu(), image)