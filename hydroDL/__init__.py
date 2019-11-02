import os
import collections

print('loading package hydroDL')


def init_path():
    """配置数据源路径，在根目录下面配置，便于各个模块都能读取该变量"""
    dir_project = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    path_gages2 = collections.OrderedDict(
        DB=os.path.join(dir_project, 'example', 'data', 'GAGES-II'),
        Out=os.path.join(dir_project, 'example', 'data', 'GAGES-II', 'rnnStreamflow'))

    path_camels = collections.OrderedDict(
        DB=os.path.join(dir_project, 'example', 'data', 'Camels'),
        Out=os.path.join(dir_project, 'example', 'data', 'Camels', 'rnnStreamflow'))
    return path_gages2, path_camels


pathGages2, pathCamels = init_path()

from . import utils
from . import data
from . import model
from . import post
