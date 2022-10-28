import detection as d
import tracker as t
import camera as c
import speed as s
import utilities as u
from tqdm import tqdm
import os
import cv2
import numpy as np


logger = u.Logger(os.path.join("Data"), "recording")
reporter = u.Reporter(os.path.join("Data", "Reports"), "recording", os.path.join("Data", "Frames"), logger, ['name', 'timestamp', 'speed(km/h)'])
