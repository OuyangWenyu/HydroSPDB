import unittest
import hydroDL
import definitions
from data import *
import os
from utils import *


class TestTrainCase(unittest.TestCase):
    config_file = definitions.CONFIG_FILE
    project_dir = definitions.ROOT_DIR
    dataset = 'gages'
    # dataset = 'camels'
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset)

    model_dict_file = os.path.join(dir_temp, 'master.json')

    data_source_dump = os.path.join(dir_temp, 'data_source.txt')
    stat_file = os.path.join(dir_temp, 'Statistics.json')
    flow_file = os.path.join(dir_temp, 'flow.npy')
    forcing_file = os.path.join(dir_temp, 'forcing.npy')
    attr_file = os.path.join(dir_temp, 'attr.npy')
    f_dict_file = os.path.join(dir_temp, 'dictFactorize.json')
    var_dict_file = os.path.join(dir_temp, 'dictAttribute.json')
    t_s_dict_file = os.path.join(dir_temp, 'dictTimeSpace.json')

    def setUp(self):
        print('setUp...读取datamodel')

        self.model_dict = unserialize_json_ordered(self.model_dict_file)

        self.source_data = unserialize_pickle(self.data_source_dump)
        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        self.stat_dict = unserialize_json(self.stat_file)
        self.data_flow = unserialize_numpy(self.flow_file)
        self.data_forcing = unserialize_numpy(self.forcing_file)
        self.data_attr = unserialize_numpy(self.attr_file)
        # dictFactorize.json is the explanation of value of categorical variables
        self.var_dict = unserialize_json(self.var_dict_file)
        self.f_dict = unserialize_json(self.f_dict_file)
        self.t_s_dict = unserialize_json(self.t_s_dict_file)
        self.data_model = DataModel(self.source_data, self.data_flow, self.data_forcing, self.data_attr, self.var_dict,
                                    self.f_dict, self.stat_dict, self.t_s_dict)

        print(self.data_model)

    def tearDown(self):
        print('tearDown...')

    def test_train(self):
        print("测试开始：")
        # 读取模型配置文件
        data_model = self.data_model
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        hydroDL.master_train(data_model)


if __name__ == '__main__':
    unittest.main()
