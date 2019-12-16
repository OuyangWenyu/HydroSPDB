"""一个处理数据的模板方法"""
from data.attribute import Attribute
from data.forcing import Forcing
from data.input_data import InputData
from data.streamflow import Streamflow


class Formatting(object):
    """数据格式化模板方法，主要分为三类数据，attributes, forcing and streamflow"""

    def __init__(self, attributes_db, forcing_db, streamflow_db):
        self.__attributes_db = attributes_db
        self.__forcing_db = forcing_db
        self.__streamflow_db = streamflow_db

    def process(self):
        attr = self.process_attr()
        forc = self.process_forc()
        flow = self.process_flow()
        return InputData(attr, forc, flow)

    def process_attr(self):
        print("processing formatting attributes")
        # 暂时设置成类，看情况，如果比较简单的数据结构，也不需要设计类
        return Attribute(self)

    def process_forc(self):
        print("processing formatting forcing")
        return Forcing(self)

    def process_flow(self):
        print("processing formatting streamflow")
        return Streamflow(self)
