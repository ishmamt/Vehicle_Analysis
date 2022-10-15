from sympy import Polygon, Line
import cv2


class Speed():
    '''
    Class to handle everything related to speed calculation.
    '''

    def __init__(self, entry_area, exit_area, deleting_line, distance_line, length):
        '''
        Constructor method for speed class.
        '''
        self.entry_area = Polygon(entry_area[0], entry_area[1], entry_area[2], entry_area[3])
        self.exit_area = Polygon(exit_area[0], exit_area[1], exit_area[2], exit_area[3])
        self.deleting_line = Line(deleting_line[0], deleting_line[1])
        self.distance_line = distance_line
        self.length = length * 0.001
        
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


    def process_frame(self, frame, tracked_objects_info, annotate, frame_count, fps):
        '''
        Process the given frame to calculate speed for all tracked objects.

            Parameters:
                frame (numpy array): Image frame to be processed.
                tracked_objects_info (list): List of tuples containing info about tracked objects.
                annotate (boolean): True if the frame needs to be annotated.
                frame_count (int): The number of frame currently being processed.
                fps (int): The FPS of the video.

            Returns:
                processed_frame (numpy array): Processed image frame. It could be annotated if specified.
        '''
        for object_info in tracked_objects_info:
            x_min, y_min, x_max, y_max, id = object_info
            object_bbox = [(x_min, y_min), (x_min + (x_max - x_min), y_min), (x_max, y_max), (x_min, y_min + (y_max - y_min))]
            speed = None

            if self.entered_the_polygon.get(id, None) is None and not self.if_intersect(object_bbox, self.entry_area):
                # The bbox with the same ID is not in the entered_the_polygon dictionary
                # and it has not crossed entry area. We do not need to calculate speed for this bbox.
                continue

            elif self.entered_the_polygon.get(id, None) is None and self.if_intersect(object_bbox, self.entry_area):
                # The bbox with the same ID is not in the entered_the_polygon dictionary
                # and it has just crossed entry area. We need to start processing this bbox.
                entry_time = frame_count / fps  # in seconds
                self.entered_the_polygon[id] = entry_time

            elif self.entered_the_polygon.get(id, None) is not None and self.if_intersect(object_bbox, self.exit_area):
                # The bbox with the same ID is in the entered_the_polygon dictionary
                # and it has just crossed exit area. We need to calculate speed and delete this ID.
                if self.speed_dictionary.get(id, None) is None:
                    # Speed for this ID had not been calculated before.
                    exit_time = frame_count / fps  # in seconds
                    speed = self.calculate_speed(self.entered_the_polygon[id], exit_time)

                    self.speed_dictionary[id] = speed

                else:
                    speed = self.speed_dictionary[id]

            elif self.entered_the_polygon.get(id, None) is not None and self.if_intersect(object_bbox, self.exit_area):
                # The bbox with the same ID is in the entered_the_polygon dictionary
                # and it has just crossed the deleting line. We need to delete this ID.
                del self.entered_the_polygon[id]
                if self.speed_dictionary.get(id, None):
                    del self.speed_dictionary[id]

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
            text = str(id)+ ": " + str(speed) + " Km/h"
            text_width, text_height = cv2.getTextSize(text, font, font_size, font_thickness)[0]
            pos_x = max((object_bbox[0] - 20),0)
            pos_y = max((object_bbox[1] - 20),0)
            cv2.rectangle(frame, (pos_x, pos_y), (pos_x + text_width, pos_y + text_height), color = (0,0,0), thickness = -1)
            cv2.putText(frame, text, (pos_x, pos_y + text_height + font_size - 1), font, font_size, font_color, font_thickness)

        return frame

    
    def calculate_speed(self, entry_time, exit_time):
        '''
        Calculates the speed of the object.

            Parameters:
                entry_time (float): The time of entry in seconds.
                exit_time (float): The time of exit in seconds.

            Returns:
                speed (float): Speed of an object in Kilometers per hour (Km/h).
        '''
        
        return round((self.length / (exit_time - entry_time)) * 3600, 3)