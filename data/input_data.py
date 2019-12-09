"""刚开始都需要获取并处理数据格式到项目数据格式，这里类的作用即定义项目数据"""


class InputData(object):
    def __init__(self, attributes, forcing, streamflow):
        self.__attributes = attributes
        self.__forcing = forcing
        self.__streamflow = streamflow

    """
    :InputData: a container of data
    """

    def get_c(self):
        """get属性数据"""
        return self.__attributes

    def get_x(self):
        """get驱动等时间序列数据"""
        return self.__forcing

    def get_y(self):
        return self.__streamflow

    def get_data_train(self):
        return self.get_x(), self.get_y(), self.get_c()
