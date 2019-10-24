import os
import collections

print('loading package hydroDL')


def init_path():
    """配置数据源路径"""
    path_gages2 = collections.OrderedDict(
        DB=os.path.join('..', os.path.sep, '..', 'example', 'data', 'GAGES-II'),
        Out=os.path.join('..', os.path.sep, '..', 'example', 'data', 'GAGES-II', 'rnnStreamflow'))

    path_camels = collections.OrderedDict(
        DB=os.path.join('..', os.path.sep, '..', 'example', 'data', 'Camels'),
        Out=os.path.join('..', os.path.sep, '..', 'example', 'data', 'Camels', 'rnnStreamflow'))
    return path_gages2, path_camels


pathGages2, pathCamels = init_path()

from . import utils
from . import data
from . import model
from . import post
