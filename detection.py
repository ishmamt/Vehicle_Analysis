import torch
import os
import cv2


class Detector():
    '''Parent class for object detection'''

    def __init__(self):
        ''' Constructor for detection class
        '''
        self.model = None


    def set_detection_model(self, git_repo="yolov5", model_type="custom", model_wights_path=os.path.join("yolov5", "models", "yolov5s.pt")):
        ''' Sets the model for detection.
        Parameters:
            git_repo: string; Reference to the git repository where the model is stored.
            model_type: string; Type of model (custom of pre-built).
            model_wights_path: string; Path to the model weights. 
        '''

        if not os.path.exists(model_wights_path):
            raise FileNotFoundError("Invalid path to model weights.")

        if model_type == "custom":
            self.model = torch.hub.load(git_repo, model_type, model_wights_path, source="local", force_reload=True)       
        else:
            raise Exception("Invalid model type.")


    def get_detection_model(self):
        ''' Fetches the detection model being used.
        Returns:
            self.model: model object; The model attribute of the face detection class.
        ''' 

        return self.model


    def get_detection_results(self, image):
        ''' Returns detection results from the model given an image.
        Parameters:
            image: numpy array; Image to run inference on.
        Returns:
            self.model(image): pytorch tensor object; The results of inference on the given image.
        '''

        return self.model(image)

    
    def get_bbox_locations(self, results, confidence_threshold=0.5):
        ''' Returns the bounding box locations of the detected ojects.
        Parameters:
            results: pytorch tensor object; The results of inference on the given image.
            confidence_threshold: float; Threshold for minimum confidence of detection.
        Returns:
            bbox_locations: list; The list of bounding box locations.
        '''

        # results.xyxy[0] is a tensor of detected objects, each arranged such as: [x_min, y_min, x_max, y_max, confidence, class]
        num_bboxes = results.xyxy[0].shape[0]
        bbox_locations = []
        for bbox in range(num_bboxes):
            if results.xyxy[0][bbox][4] > confidence_threshold:
                x_min = int(results.xyxy[0][bbox][0].item())
                y_min = int(results.xyxy[0][bbox][1].item()) + 10
                x_max = int(results.xyxy[0][bbox][2].item())
                y_max = int(results.xyxy[0][bbox][3].item())

                bbox_locations.append((y_min, x_max, y_max, x_min))
        
        return bbox_locations

    
    def annotate(self, image, bbox_locations, color_primary=(72, 72, 255), box_width=2):
        ''' Returns the image with the bounding boxes annotated.
        Parameters:
            image: numpy array; The input to be annotated image.
            bbox_locations: list; The list of bounding box locations.
            color_primary: tuple; Color of the bounding boxes.
            box_width: int; Width of the bounding boxes.
        Returns:
            image: numpy array; The annotated image.
        '''
        
        for (top, right, bottom, left) in bbox_locations: 
            cv2.rectangle(image, (left, top), (right, bottom), color_primary, box_width)
        
        return image

    
    def detect(self, image):
        ''' Takes an image and returns the image detected objects annotated.
        Parameters:
            image: numpy array; The input to be annotated image.
        Returns:
            image: numpy array; The annotated image.
        '''

        results = self.get_detection_results(image)
        bbox_locations = self.get_bbox_locations(results)
        return self.annotate(image, bbox_locations)


class VehicleDetector(Detector):
    def __init__(self, git_repo="yolov5", model_type="custom", model_wights_path=os.path.join("yolov5", "models", "yolov5s.pt")):
        ''' Vehicle detector class. Inherits from Detector class.
        Parameters:
            git_repo: string; Reference to the git repository where the model is stored.
            model_type: string; Type of model (custom of pre-built).
            path: string; Path to the model weights.
        '''

        super().__init__()
        self.set_detection_model()