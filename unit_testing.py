import detection as d
import tracker as t
import camera as c
import speed as s
import utilities as u
from tqdm import tqdm
import os
import cv2
import numpy as np
from sympy import Polygon


# logger = u.Logger(os.path.join("Data"), "recording")
# reporter = u.Reporter(os.path.join("Data", "Reports"), "recording", os.path.join("Data", "Frames"), logger, ['name', 'timestamp', 'speed(km/h)'])
# p1 = (0, 0)
# p2 = (0, 5)
# p3 = (6.5, 5.5)

# p1 = np.asarray(p1)
# p2 = np.asarray(p2)
# p3 = np.asarray(p3)

# # print(np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2-p1))



entry_area = [(423, 288), (753, 297), (768, 334), (414, 321)]
exit_area = [(320, 535), (880, 530), (890, 560), (315, 562)]
a = Polygon(entry_area[0], entry_area[1], entry_area[2], entry_area[3])
# # a = np.array(a.vertices[0:])

# # print(a)



def shortest_distance(entry_area, object_bbox):
    '''
    Returns the shortest distance between a point an a line.
    
        Parameters:
            object_bbox (list): List of tuples denoting the bounding box such as: [(xmin, ymin), (xmin + w, ymin), (xmax, ymax), (xmin, ymin + h)].
            
        Returns:
            distance (float): The distance value.
    '''
    p1 = np.asarray(tuple(entry_area.vertices[0]), dtype=np.float32)
    p2 = np.asarray(tuple(entry_area.vertices[1]), dtype=np.float32)
    x1_y1, x2_y2 = object_bbox[-2: ]
    p3 = ((x1_y1[0] + x2_y2[0]) / 2, (x1_y1[1] + x2_y2[1]) / 2)
    p3 = np.asarray(p3, dtype=np.float32)
    
    return int(np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2-p1))


print(shortest_distance(a, list(reversed(exit_area))))

