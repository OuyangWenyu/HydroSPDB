import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from data import CamelsSource, DataModel
from data.data_input import CamelsModel
from utils import hydro_time
from utils.hydro_math import copy_attr_array_in2d, concat_two_3darray


class CamelsModels(object):
    """the data model for CAMELS dataset"""

    def __init__(self, config_data):
        # prepare training data
        t_train = config_data.model_dict["data"]["tRangeTrain"]
        t_test = config_data.model_dict["data"]["tRangeTest"]
        t_train_test = [t_train[0], t_test[1]]
        source_data = CamelsSource(config_data, t_train_test)
        # create datamodel as input for the model
        data_model = CamelsModel(source_data)
        self.data_model_train, self.data_model_test = CamelsModel.data_models_of_train_test(data_model, t_train, t_test)
