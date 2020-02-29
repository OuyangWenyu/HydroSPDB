import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from data import CamelsSource, DataModel
from data.data_input import CamelsModel
from utils import hydro_time
from utils.hydro_math import copy_attr_array_in2d, concat_two_3darray


class SimNatureFlowInput(object):
    def __init__(self, data_source):
        self.data_source = data_source
        self.data_input = data_source.read_natural_inflow()

    def get_data_inflow(self, rm_nan=True):
        """径流数据读取及归一化处理，会处理成三维，最后一维长度为1，表示径流变量"""
        data = self.data_input
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        # transform x to 3d, the final dim's length is the seq_length
        seq_length = self.data_source.model_data.data_source.data_config.model_dict["model"]["seqLength"]
        data_inflow = np.zeros([data.shape[0], data.shape[1] - seq_length + 1, seq_length])
        for i in range(data_inflow.shape[1]):
            data_inflow[:, i, :] = data[:, i:i + seq_length]
        return data_inflow

    def load_data(self, model_dict):
        """transform x to 3d, the final dim's length is the seq_length, add forcing with natural flow"""

        def cut_data(temp_x, temp_rm_nan, temp_seq_length):
            """cut to size same as inflow's"""
            temp = temp_x[:, temp_seq_length - 1:, :]
            if temp_rm_nan:
                temp[np.where(np.isnan(temp))] = 0
            return temp

        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        q = self.get_data_inflow(rm_nan=rm_nan_x)
        x, y, c = self.data_source.model_data.load_data(model_dict)
        seq_length = model_dict["model"]["seqLength"]

        if seq_length > 1:
            x = cut_data(x, rm_nan_x, seq_length)
            y = cut_data(y, rm_nan_y, seq_length)
        qx = np.array([np.concatenate((q[j], x[j]), axis=1) for j in range(q.shape[0])])
        return qx, y, c


class CamelsModels(object):
    """the data model for CAMELS dataset"""

    def __init__(self, config_data):
        # 准备训练数据
        t_train = config_data.model_dict["data"]["tRangeTrain"]
        t_test = config_data.model_dict["data"]["tRangeTest"]
        t_train_test = [t_train[0], t_test[1]]
        source_data = CamelsSource(config_data, t_train_test)
        # 构建输入数据类对象
        data_model = CamelsModel(source_data)
        self.data_model_train, self.data_model_test = CamelsModel.data_models_of_train_test(data_model, t_train, t_test)
