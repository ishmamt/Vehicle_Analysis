import os
from datetime import datetime
import csv
import cv2


class FrameSkipper():
    '''
    Class for managing skipping of frames.

        Attributes:
            skipped_frame_count (int): Number of frames skipped.
            frames_skipped_before_processing (int): Number of frames to skip before processing.
    '''

    def __init__(self, frames_skipped_before_processing=6):
        '''
        Constructor for the FrameSkipper class.
        
            Parameters:
                frames_skipped_before_processing (int): Number of frames to skip before processing.
        '''
        self.skipped_frame_count = 0
        self.frames_skipped_before_processing = frames_skipped_before_processing
    

    def increment_skipped_frame_count(self):
        '''
        Increments skipped frame count.
        '''
        self.skipped_frame_count += 1
    

    def reset_skipped_frame_count(self):
        '''
        Resets skipped frame count to 0.
        '''
        if self.skipped_frame_count == self.frames_skipped_before_processing:
            self.skipped_frame_count = 0

    
    def if_process_frame(self):
        '''
        Check to see if the current frame will be processed.

            Returns:
                process_frame (boolean): Returns a true or false value based on if the frame will be processed.
        '''

        return self.skipped_frame_count == 0


class Logger():
    '''
    Class to handle logging.

        Attributes:
            log_path (str): Path to generate the log file.
            video_name (str): Name of the video file to be processed.
            importance_levels (list): List of importance levels. They are: DEBUG, INFO, WARNING, ERROR, CRITICAL
            datetime_format (str): Datetime format string. By default they are: Day-Month-Year  Hour:Minute:Second.
    '''

    def __init__(self, log_path, video_name):
        '''
        Constructor method to intialize a logger class.

        Parameters:
            log_path (str): Path to generate the log file.
            video_name (str): Name of the video file to be processed.

        Returns:
            logger (logger object): The logger object.
        '''
        if not os.path.exists(log_path):
            print(f"Log directory does not exist. Creating log directory: {log_path}")
            os.makedirs(log_path)

        self.log_path = log_path
        self.video_name = video_name
        self.importance_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.datetime_format = "%d-%m-%Y  %H:%M:%S"


    def configure_log_message(self, level, message):
        '''
        Configure a message to add to log file.

            Parameters:
                level (string): Importance level of the message. Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
                message (string): Message to add to the log file.

            Returns:
                configured_message (string): Message configured to specification.
        '''
        if level not in self.importance_levels:
            raise Exception(f"Invalid importance level of log message: {level}. It should be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")

        return f"{level} | {datetime.now().strftime(self.datetime_format)} | {message}"

    
    def write_to_log(self, message):
        '''
        Method to open and append a message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        with open(os.path.join(self.log_path, f"{self.video_name}.log"), "a") as log_file:
            log_file.write(f"{message}\n")

    
    def debug(self, message):
        '''
        Method for adding a debug message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.write_to_log(self.configure_log_message("DEBUG", message))

    
    def info(self, message):
        '''
        Method for adding a info message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.write_to_log(self.configure_log_message("INFO", message))

    
    def warning(self, message):
        '''
        Method for adding a warning message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.write_to_log(self.configure_log_message("WARNING", message))

    
    def error(self, message):
        '''
        Method for adding a error message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.write_to_log(self.configure_log_message("ERROR", message))


    def critical(self, message):
        '''
        Method for adding a critical message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.write_to_log(self.configure_log_message("CRITICAL", message))
        
        
class Reporter():
    '''
    Class for generating reports.
    
        Attributes:
            report_path (str): Path to generate the report file.
            video_name (str): Name of the video file to be processed.
            image_directory (str): Directory to save images.
            logger (Logger object): Logger object for logging.
            header (list): List of headings for the report file.
    '''
    
    def __init__(self, report_path, video_name, image_directory, logger, header):
        '''
        Constructor method to intialize a reporter class.

        Parameters:
            report_path (str): Path to generate the report file.
            video_name (str): Name of the video file to be processed.
            image_directory (str): Directory to save images.
            logger (Logger object): Logger object for logging.
            header (list): List of headings for the report file.
        '''
        if not os.path.exists(report_path):
            logger.info(f"Results directory does not exist. Creating results directory: {report_path}")
            os.makedirs(report_path)
            
        if not os.path.exists(image_directory):
            logger.info(f"Image directory does not exist. Creating image directory: {os.path.join(image_directory, video_name)}")
            os.makedirs(os.path.join(image_directory, video_name))

        self.report_path = report_path
        self.image_directory = image_directory
        self.video_name = video_name
        self.logger = logger
        self.header = header
        
        self.create_report()  
    
    
    def create_report(self):
        '''
        Creates the initial report file.
        '''
        with open(os.path.join(self.report_path, f"{self.video_name}_report.csv"), 'a', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.header)
            
            
    def add_to_report(self, frame, id, speed, bbox, time):
        '''
        Adding a data point to the report file.
        
            Parameters:
                frame (numpy array): Video frame.
                id (int): ID of the tracekd object.
                speed (float): Speed of the object.
                bbox (tuple): Object bounding box.
                time (float): The current time of the video.
        '''
        xmin, ymin, xmax, ymax = bbox
        cropped_frame = frame[ymin: ymax, xmin: xmax]
        cv2.imwrite(os.path.join(self.image_directory, self.video_name, f"{id}.png"), cropped_frame)

        # if int(time / 60) == 60:
        #     time = time % 60
        time2 = f"{int(time / 60)}:{int(time % 60)}"
        
        self.add_a_row_to_report([str(id), time2, str(speed)])
        
        
    def add_a_row_to_report(self, row):
        '''
        Method to add a row to the report file.
        
            Parameters:
                row (list): The data point to write tot he report file.
        '''
        with open(os.path.join(self.report_path, f"{self.video_name}_report.csv"), 'a', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
