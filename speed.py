from sympy import Polygon
import cv2
import numpy as np


class Speed():
    '''
    Class to handle everything related to speed calculation.
    
        Attributes:
            area (Polygon): Processing area.
            area_asList (list): Processing area as a list.
            deleting_line (Polygon): Deleting line.
            length (float): Length of the processing area.
            logger (Logger object): Logger object for logging.
    '''

    def __init__(self, entry_area, exit_area, deleting_line, length, logger):
        '''
        Constructor method for speed class.
        
            Parameters:
                area (Polygon): Processing area.
                area_asList (list): Processing area as a list.
                deleting_line (Polygon): Deleting line.
                length (float): Length of the processing area.
                logger (Logger object): Logger object for logging.`
        '''
        self.entry_area = Polygon(entry_area[0], entry_area[1], entry_area[2], entry_area[3])
        self.exit_area = Polygon(exit_area[0], exit_area[1], exit_area[2], exit_area[3])
        self.deleting_line = Polygon(deleting_line[0], deleting_line[1], deleting_line[2], deleting_line[3])
        self.length = length * 0.001
        self.logger = logger
        self.pixel_distance = int(self.shortest_distance(list(reversed(exit_area))))
        self.pixel_ratio = self.length / self.pixel_distance
        
        self.entered_the_polygon = {}  # {Object ID: entry_time}
        self.speed_dictionary = {}  # {Object ID: speed}


    def if_intersect(self, object_bbox, area):
        '''
        Check to see if object bounding box intersects with the given area.

            Parameters:
                object_bbox (list): List of tuples denoting the bounding box such as: [(xmin, ymin), (xmin + w, ymin), (xmax, ymax), (xmin, ymin + h)]
                area (list): List of tuples denoting an area such as: [(xmin, ymin), (xmin + w, ymin), (xmax, ymax), (xmin, ymin + h)]

            Returns:
                if_intersect (boolean): True if object bounding box intersects with the given area.
        '''
        bbox_polygon = Polygon(object_bbox[0], object_bbox[1], object_bbox[2], object_bbox[3])
        
        return len(bbox_polygon.intersection(area)) > 0

    
    def if_inside(self, object_bbox_center, area):
        '''
        Check to see if object bounding box is inside an area.
        
            Parameters:
                object_bbox (list): List of tuples denoting the bounding box such as: [(xmin, ymin), (xmin + w, ymin), (xmax, ymax), (xmin, ymin + h)]
                area (list): List of tuples denoting an area such as: [(xmin, ymin), (xmin + w, ymin), (xmax, ymax), (xmin, ymin + h)]
                
            Returns:
                if_inside (boolean): True if object bounding box is inside the given area.
        '''
        result = cv2.pointPolygonTest(np.array(area, np.int32),(int(object_bbox_center[0]), int(object_bbox_center[1])), False)
        return result >= 0.0
    
    
    def shortest_distance(self, object_bbox):
        '''
        Returns the shortest distance between a point an a line.
        
            Parameters:
                object_bbox (list): List of tuples denoting the bounding box such as: [(xmin, ymin), (xmin + w, ymin), (xmax, ymax), (xmin, ymin + h)].
                
            Returns:
                distance (float): The distance value.
        '''
        p1 = np.asarray(tuple(self.entry_area.vertices[0]), dtype=np.float32)
        p2 = np.asarray(tuple(self.entry_area.vertices[1]), dtype=np.float32)
        x1_y1, x2_y2 = object_bbox[-2: ]
        p3 = ((x1_y1[0] + x2_y2[0]) / 2, (x1_y1[1] + x2_y2[1]) / 2)
        p3 = np.asarray(p3, dtype=np.float32)
        
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


    def process_frame(self, frame, tracked_objects_info, annotate, frame_count, fps, current_timestamp, reporter):
        '''
        Process the given frame to calculate speed for all tracked objects.

            Parameters:
                frame (numpy array): Image frame to be processed.
                tracked_objects_info (list): List of tuples containing info about tracked objects.
                annotate (boolean): True if the frame needs to be annotated.
                frame_count (int): The number of frame currently being processed.
                fps (int): The FPS of the video.
                current_timestamp (string): The current timestamp of the video.
                reporter (Reporter object): Reporter object for adding to reports.

            Returns:
                processed_frame (numpy array): Processed image frame. It could be annotated if specified.
        '''
        for object_info in tracked_objects_info:
            x_min, y_min, x_max, y_max, id = object_info
            object_bbox = [(x_min, y_min), (x_min + (x_max - x_min), y_min), (x_max, y_max), (x_min, y_min + (y_max - y_min))]
            speed = None
            # self.logger.debug(f"Object ID: {id} being tracked.")

            if self.entered_the_polygon.get(id, None) is None and not self.if_intersect(object_bbox, self.entry_area):
                # The bbox with the same ID is not in the entered_the_polygon dictionary
                # and it has not crossed entry area. We do not need to calculate speed for this bbox.
                pass

            elif self.entered_the_polygon.get(id, None) is None and self.if_intersect(object_bbox, self.entry_area):
                # The bbox with the same ID is not in the entered_the_polygon dictionary
                # and it has just crossed entry area. We need to start processing this bbox.
                entry_time = frame_count / fps  # in seconds
                self.entered_the_polygon[id] = [entry_time, object_bbox]
                self.logger.debug(f"Object with ID: {id} crossed the entry line. {self.entered_the_polygon} {self.speed_dictionary}")

            elif self.entered_the_polygon.get(id, None) is not None and self.if_intersect(object_bbox, self.exit_area):
                # The bbox with the same ID is in the entered_the_polygon dictionary
                # and it has just crossed exit area. We need to calculate speed.
                if self.speed_dictionary.get(id, None) is None:
                    # Speed for this ID had not been calculated before.
                    exit_time = frame_count / fps  # in seconds
                    pixel_distance_from_bbox = self.shortest_distance(self.entered_the_polygon[id][1])
                    speed = self.calculate_speed(self.entered_the_polygon[id][0], exit_time, pixel_distance_from_bbox)

                    self.speed_dictionary[id] = speed
                    self.logger.debug(f"Object with ID: {id} crossed the exit line: {exit_time}. entry_time: {self.entered_the_polygon} {self.speed_dictionary}")
                    reporter.add_to_report(frame, id, speed, (x_min, y_min, x_max, y_max), current_timestamp)

                else:
                    speed = self.speed_dictionary[id]

            if self.entered_the_polygon.get(id, None) is not None and self.if_intersect(object_bbox, self.deleting_line):
                # The bbox with the same ID is in the entered_the_polygon dictionary
                # and it has just crossed the deleting line. We need to delete this ID.
                del self.entered_the_polygon[id]
                if self.speed_dictionary.get(id, None):
                    del self.speed_dictionary[id]
                
                self.logger.debug(f"Object with ID: {id} crossed the delete line. {self.entered_the_polygon} {self.speed_dictionary}")

            if annotate:
                frame = self.annotate(frame, [x_min, y_min, x_max, y_max], speed, id)

        return frame


    def annotate(self, frame, object_bbox, speed, id, color_primary=(72, 72, 255), 
                box_width=2, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=1, 
                font_color=(255, 255, 255), font_thickness = 1):
        '''
        Annotates the frame.

            Parameters:
                frame (numpy array): The input frame.
                object_bbox (list): The list of bounding box location.
                color_primary (tuple): Color of the bounding boxe.
                box_width (int): Width of the bounding boxe.

            Returns:
                frame (numpy array): The annotated frame.
        '''
        cv2.rectangle(frame, (object_bbox[0], object_bbox[1]), (object_bbox[2], object_bbox[3]), color_primary, box_width)

        if speed:
            text = "ID: " + str(id)+ " Speed: " + str(speed) + " Km/h"
            text_width, text_height = cv2.getTextSize(text, font, font_size, font_thickness)[0]
            pos_x = max((object_bbox[0] - 20),0)
            pos_y = max((object_bbox[1] - 20),0)
            cv2.rectangle(frame, (pos_x, pos_y), (pos_x + text_width, pos_y + text_height), color = (0,0,0), thickness = -1)
            cv2.putText(frame, text, (pos_x, pos_y + text_height + font_size - 1), font, font_size, font_color, font_thickness)

        return frame

    
    def calculate_speed(self, entry_time, exit_time, pixel_distance_from_bbox):
        '''
        Calculates the speed of the object.

            Parameters:
                entry_time (float): The time of entry in seconds.
                exit_time (float): The time of exit in seconds.

            Returns:
                speed (float): Speed of an object in Kilometers per hour (Km/h).
        '''
        
        return round(((self.length - (pixel_distance_from_bbox * self.pixel_ratio)) / (exit_time - entry_time)) * 3600, 3)