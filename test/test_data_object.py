"""模块测试"""
import os
import unittest
from data import *
from utils import *


class MyTestCase(unittest.TestCase):
    config_file = r"../data/config.ini"
    data_source_dump = '/home/owen/Documents/Code/hydro-anthropogenic-lstm/example/temp/gages/data_source.txt'

    def test_data_source(self):
        config_file = self.config_file

        # 读取模型配置文件
        opt_train, opt_data, opt_model, opt_loss = init_model_param(config_file)

        # 准备训练数据
        source_data = SourceData(config_file, opt_data.get("tRangeTrain"), ['1980-01-01', '2015-01-01'])
        # 序列化保存对象
        dir_temp = source_data.all_configs["temp_dir"]
        if not os.path.isdir(dir_temp):
            os.mkdir(dir_temp)
        my_file = os.path.join(dir_temp, 'data_source.txt')
        serialize_pickle(source_data, my_file)

    def test_read_data_source_temp(self):
        d = unserialize_pickle(self.data_source_dump)
        print(d)

    def test_data_model(self):
        source_data = unserialize_pickle(self.data_source_dump)
        print(source_data)
        data_model = DataModel(source_data)
        print(data_model)
        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        data_flow = data_model.data_flow
        data_forcing = data_model.data_forcing
        data_attr = data_model.data_attr
        stat_dict = data_model.stat_dict
        # 序列化保存对象
        dir_temp = source_data.all_configs["temp_dir"]
        if not os.path.isdir(dir_temp):
            os.mkdir(dir_temp)
        stat_file = os.path.join(dir_temp, 'Statistics.json')
        flow_file = os.path.join(dir_temp, 'flow')
        forcing_file = os.path.join(dir_temp, 'forcing')
        attr_file = os.path.join(dir_temp, 'attr')
        serialize_json(stat_dict, stat_file)
        serialize_numpy(data_flow, flow_file)
        serialize_numpy(data_forcing, forcing_file)
        serialize_numpy(data_attr, attr_file)


if __name__ == '__main__':
    unittest.main()
