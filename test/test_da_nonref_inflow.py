import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        print('setUp...set datamodel')
        config_file = definitions.CONFIG_FILE

    def test_da(self):
        """make data integration for nonref regions"""
        # 读取模型配置文件
        config_data = GagesConfig(config_file)
        # 准备训练数据
        source_data = GagesSource(config_data, config_data.model_dict["data"]["tRangeTrain"])
        # make input dataset for data integration
        data_model = DataLoaderDa(source_data)
        # train model
        master_train(data_model)

    def test_da_with_nature_flow(self):
        """use natural flow generated by allref regions"""
        config_data = SimNatureFlowConfig(config_file)
        source_data = SimNatureFlowSource(config_data, config_data.model_dict["data"]["tRangeTrain"])
        dataModel = DataLoaderDa(source_data)
        hydroDL.master_train(data_model)


if __name__ == '__main__':
    unittest.main()
