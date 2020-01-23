import unittest


class MyTestCase(unittest.TestCase):
    def test_stacked_lstm(self):
        """try to input data from nonref regions to a stacked lstm to see what will happen"""
        config_file = definitions.CONFIG_FILE
        # 读取模型配置文件
        config_data = GagesConfig(config_file)
        # 准备训练数据
        source_data = GagesSource(config_data, config_data.model_dict["data"]["tRangeTrain"])
        # 构建输入数据类对象
        data_model = DataModel(source_data)
        # 进行模型训练
        # train model
        master_train(data_model)

if __name__ == '__main__':
    unittest.main()
