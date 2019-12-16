"""一个处理数据的模板方法"""
import datetime as dt
import json
import os
from collections import OrderedDict

import numpy as np

from data.attribute import Attribute
from data.forcing import Forcing
from data.input_data import InputData
from data.streamflow import Streamflow
from hydroDL import utils


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


def wrap_master(out, optData, optModel, optLoss, optTrain):
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain)
    return mDict


def read_master_file(out):
    m_file = os.path.join(out, 'master.json')
    with open(m_file, 'r') as fp:
        m_dict = json.load(fp, object_pairs_hook=OrderedDict)
    print('read master file ' + m_file)
    return m_dict


def write_master_file(m_dict):
    out = m_dict['out']
    if not os.path.isdir(out):
        os.makedirs(out)
    m_file = os.path.join(out, 'master.json')
    with open(m_file, 'w') as fp:
        json.dump(m_dict, fp, indent=4)
    print('write master file ' + m_file)
    return out


def namePred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
    if not os.path.exists(out):
        os.makedirs(out)
    if not os.path.exists(os.path.join(out, 'master.json')):
        return ['None']
    mDict = read_master_file(out)
    target = mDict['data']['target']
    if type(target) is not list:
        target = [target]
    nt = len(target)
    lossName = mDict['loss']['name']
    if epoch is None:
        epoch = mDict['train']['nEpoch']

    fileNameLst = list()
    for k in range(nt):
        testName = '_'.join(
            [subset, str(tRange[0]),
             str(tRange[1]), 'ep' + str(epoch)])
        fileName = '_'.join([target[k], testName])
        fileNameLst.append(fileName)
        if lossName == 'hydroDL.model.crit.SigmaLoss':
            fileName = '_'.join([target[k] + 'SigmaX', testName])
            fileNameLst.append(fileName)

    # sum up to file path list
    filePathLst = list()
    for fileName in fileNameLst:
        if suffix is not None:
            fileName = fileName + '_' + suffix
        filePath = os.path.join(out, fileName + '.csv')
        filePathLst.append(filePath)
    return filePathLst


def load_data(opt_data):
    if eval(opt_data['name']) is app.streamflow.data.gages2.DataframeGages2:
        df = app.streamflow.data.gages2.DataframeGages2(
            subset=opt_data['subset'],
            t_range=opt_data['tRange'])
    elif eval(opt_data['name']) is app.streamflow.data.camels.DataframeCamels:
        df = app.streamflow.data.camels.DataframeCamels(
            subset=opt_data['subset'], t_range=opt_data['tRange'])
    else:
        raise Exception('unknown database')
    x = df.get_data_ts(
        var_lst=opt_data['varT'],
        do_norm=opt_data['doNorm'][0],
        rm_nan=opt_data['rmNan'][0])
    y = df.get_data_obs(
        do_norm=opt_data['doNorm'][1], rm_nan=opt_data['rmNan'][1])
    c = df.get_data_const(
        var_lst=opt_data['varC'],
        do_norm=opt_data['doNorm'][0],
        rm_nan=opt_data['rmNan'][0])
    if opt_data['daObs'] > 0:
        nday = opt_data['daObs']
        sd = utils.time.t2dt(
            opt_data['tRange'][0]) - dt.timedelta(days=nday)
        ed = utils.time.t2dt(
            opt_data['tRange'][1]) - dt.timedelta(days=nday)
        if eval(opt_data['name']) is app.streamflow.data.gages2.DataframeGages2:
            df = app.streamflow.data.gages2.DataframeGages2(subset=opt_data['subset'], tRange=[sd, ed])
        elif eval(opt_data['name']) is app.streamflow.data.camels.DataframeCamels:
            df = app.streamflow.data.camels.DataframeCamels(subset=opt_data['subset'], tRange=[sd, ed])
        obs = df.get_data_obs(do_norm=opt_data['doNorm'][1], rm_nan=True)
        x = np.concatenate([x, obs], axis=2)
    return df, x, y, c