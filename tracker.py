import os
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class Tracker():
    def __init__(self):
        ''' Constructor for Tracker class.
        '''

        self.model = None

    
    def set_tracker_model(self, model_name="osnet_x0_25", config_path=os.path.join("deep_sort", "configs", "deep_sort.yaml")):
        ''' Sets the model for tracking.
        Parameters:
            model_name: string; Name of the model.
            config_path: string; Path to the config.yaml file.
        '''

        if not os.path.exists(config_path):
            raise FileNotFoundError("Invalid path to config file.")

        config = get_config()
        config.merge_from_file(config_path)

        self.model = DeepSort(model_name, max_dist=config.DEEPSORT.MAX_DIST, max_iou_distance=config.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=config.DEEPSORT.MAX_AGE, n_init=config.DEEPSORT.N_INIT, nn_budget=config.DEEPSORT.NN_BUDGET, use_cuda=True)


class VehicleTracker(Tracker):
    def __init__(self):
        pass