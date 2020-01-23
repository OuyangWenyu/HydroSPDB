import unittest


class MyTestCase(unittest.TestCase):
    def test_some_reservoirs(self):
        """choose some small reservoirs randomly to train and test"""
        config_file = definitions.CONFIG_FILE
        # 读取模型配置文件
        config_data = GagesConfig(config_file)
        # choose some ids randomly
        screen_id = rand_choose(config_data)
        # update config
        hidden_size = 50  # try small hidden_size
        config_data = config_data.update(screen_id, hidden_size)
        # 准备训练数据
        source_data = GagesSource(config_data, config_data.model_dict["data"]["tRangeTrain"])
        # 构建输入数据类对象
        data_model = DataModel(source_data)
        # train model
        master_train(data_model)

if __name__ == '__main__':
    unittest.main()
