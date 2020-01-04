import os
import unittest
import hydroDL
from data import DataModel, GagesSource, GagesConfig
from utils import unserialize_json, serialize_pickle, unserialize_pickle, serialize_json, serialize_numpy, \
    unserialize_json_ordered, unserialize_numpy


class TestForecastCase(unittest.TestCase):
    config_file = r"../data/config.ini"
    root = os.path.expanduser('~')
    project_dir = 'Documents/Code/hydro-anthropogenic-lstm'
    dir_db = os.path.join(root, project_dir, 'example/data/gages')
    dir_out = os.path.join(root, project_dir, 'example/output/gages')
    dir_temp = os.path.join(root, project_dir, 'example/temp/gages')
    model_dict_file = os.path.join(dir_temp, 'master.json')
    data_source_test_file = os.path.join(dir_temp, 'data_source_test.txt')

    stat_file = os.path.join(dir_temp, 'Statistics_test.json')
    flow_file = os.path.join(dir_temp, 'flow_test')
    flow_npy_file = os.path.join(dir_temp, 'flow_test.npy')
    forcing_file = os.path.join(dir_temp, 'forcing_test')
    forcing_npy_file = os.path.join(dir_temp, 'forcing_test.npy')
    attr_file = os.path.join(dir_temp, 'attr_test')
    attr_npy_file = os.path.join(dir_temp, 'attr_test.npy')
    f_dict_file = os.path.join(dir_temp, 'dictFactorize_test.json')
    var_dict_file = os.path.join(dir_temp, 'dictAttribute_test.json')
    t_s_dict_file = os.path.join(dir_temp, 'dictTimeSpace_test.json')

    flow_pred_file = os.path.join(dir_temp, 'flow_pred')
    flow_obs_file = os.path.join(dir_temp, 'flow_obs')

    def test_data_source_test(self):
        config_data = GagesConfig(self.config_file)
        # 准备训练数据
        source_data = GagesSource(config_data, config_data.model_dict["data"]["tRangeTest"])
        # 序列化保存对象
        serialize_pickle(source_data, self.data_source_test_file)

    def test_data_model_test(self):
        source_data = unserialize_pickle(self.data_source_test_file)
        data_model = DataModel(source_data)
        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        serialize_json(data_model.stat_dict, self.stat_file)
        serialize_numpy(data_model.data_flow, self.flow_file)
        serialize_numpy(data_model.data_forcing, self.forcing_file)
        serialize_numpy(data_model.data_attr, self.attr_file)
        # dictFactorize.json is the explanation of value of categorical variables
        serialize_json(data_model.f_dict, self.f_dict_file)
        serialize_json(data_model.var_dict, self.var_dict_file)
        serialize_json(data_model.t_s_dict, self.t_s_dict_file)

    def test_forecast(self):
        source_data = unserialize_pickle(self.data_source_test_file)
        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        stat_dict = unserialize_json(self.stat_file)
        data_flow = unserialize_numpy(self.flow_npy_file)
        data_forcing = unserialize_numpy(self.forcing_npy_file)
        data_attr = unserialize_numpy(self.attr_npy_file)
        # dictFactorize.json is the explanation of value of categorical variables
        var_dict = unserialize_json(self.var_dict_file)
        f_dict = unserialize_json(self.f_dict_file)
        t_s_dict = unserialize_json(self.t_s_dict_file)
        data_model_test = DataModel(source_data, data_flow, data_forcing, data_attr, var_dict, f_dict, stat_dict,
                                    t_s_dict)
        pred, obs = hydroDL.master_test(data_model_test)
        print(pred)
        print(obs)
        serialize_numpy(pred, self.flow_pred_file)
        serialize_numpy(obs, self.flow_obs_file)


if __name__ == '__main__':
    unittest.main()
