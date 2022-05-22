import torch
import os
import cv2
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.dataloaders import LoadImages
import timeit
import csv
from sympy import Point, Polygon, Line
from tqdm import tqdm


model = torch.hub.load("yolov5", "custom", source="local", path="yolov5/yolov5s.pt", force_reload=True)
img = cv2.imread("test.jpg")

results = model(img)

results.show()