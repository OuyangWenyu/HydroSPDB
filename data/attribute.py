"""属性值定义一个属性类，主要是一个属性类型和属性值的dict"""


class Attribute(object):
    def __init__(self, a):
        self.__a = a
