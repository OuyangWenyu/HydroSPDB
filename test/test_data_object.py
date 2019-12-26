"""模块测试"""
import os
import unittest
from data import *
import pickle


class MyTestCase(unittest.TestCase):
    config_file = r"../data/config.ini"
    data_source_dump = '/home/owen/Documents/Code/hydro-anthropogenic-lstm/example/temp/gages/dump.txt'

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
        f = open(os.path.join(dir_temp, 'dump.txt'), 'wb')
        pickle.dump(source_data, f)
        f.close()

    def test_read_data_source_temp(self):
        f = open(self.data_source_dump, 'rb')
        d = pickle.load(f)
        print(d)
        f.close()

    def test_data_model(self):
        f = open(self.data_source_dump, 'rb')
        source_data = pickle.load(f)
        print(source_data)
        f.close()
        data_model = DataModel(source_data)
        print(data_model)


if __name__ == '__main__':
    unittest.main()
