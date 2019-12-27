import unittest
import hydroDL
import os
import utils
import numpy as np

from data import DataModel
from utils import *


class MyTestCase(unittest.TestCase):
    config_file = r"../data/config.ini"
    dir_temp = '/home/owen/Documents/Code/hydro-anthropogenic-lstm/example/temp/gages'
    data_source_dump = os.path.join(dir_temp, 'data_source.txt')
    stat_file = os.path.join(dir_temp, 'Statistics.json')
    flow_file = os.path.join(dir_temp, 'flow.npy')
    forcing_file = os.path.join(dir_temp, 'forcing.npy')
    attr_file = os.path.join(dir_temp, 'attr.npy')
    f_dict_file = os.path.join(dir_temp, 'dictFactorize.json')
    var_dict_file = os.path.join(dir_temp, 'dictAttribute.json')

    def test_read_data_model(self):
        source_data = unserialize_pickle(self.data_source_dump)

        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        stat_dict = unserialize_json(self.stat_file)
        data_flow = unserialize_numpy(self.flow_file)
        data_forcing = unserialize_numpy(self.forcing_file)
        data_attr = unserialize_numpy(self.attr_file)
        # dictFactorize.json is the explanation of value of categorical variables
        var_dict = unserialize_json(self.var_dict_file)
        f_dict = unserialize_json(self.f_dict_file)
        data_model = DataModel(source_data, data_flow, data_forcing, data_attr, var_dict, f_dict, stat_dict)
        print(data_model)

    def test_train(self):
        rootDB = hydroDL.pathSMAP['DB_L3_NA']
        nEpoch = 5
        outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'cnnCond')
        ty1 = [20150402, 20160401]
        ty2 = [20160401, 20170401]
        ty12 = [20150402, 20170401]
        ty3 = [20170401, 20180401]

        doLst = list()
        # doLst.append('trainCnn')
        # doLst.append('trainLstm')
        doLst.append('testCnn')
        doLst.append('testLstm')
        doLst.append('post')

        if 'trainLstm' in doLst:
            df = app.streamflow.data.dbCsv.DataframeCsv(
                rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
            x = df.getData(
                varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
            y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
            nx = x.shape[-1]
            ny = 1
            model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=64)
            lossFun = crit.RmseLoss()
            model = train.model_train(
                model, x, y, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
            modelName = 'lstm_y1'
            train.model_save(outFolder, model, nEpoch, modelName=modelName)

        if 'trainCnn' in doLst:
            dfc = app.streamflow.data.dbCsv.DataframeCsv(
                rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
            xc = dfc.getData(
                varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
            yc = dfc.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
            yc[:, :, 0] = utils.interpNan(yc[:, :, 0])
            c = np.concatenate((yc, xc), axis=2)
            df = app.streamflow.data.dbCsv.DataframeCsv(
                rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
            x = df.getData(
                varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
            y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
            nx = x.shape[-1]
            ny = 1
            for opt in range(1, 4):
                model = rnn.LstmCnnCond(
                    nx=nx, ny=ny, ct=365, hiddenSize=64, cnnSize=32, opt=opt)
                lossFun = crit.RmseLoss()
                model = train.model_train(model, (x, c), y, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
                modelName = 'cnn' + str(opt) + '_y1y1'
                train.model_save(outFolder, model, nEpoch, modelName=modelName)

        ypLst = list()
        df = app.streamflow.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
        yT = df.getData(varT='SMAP_AM', doNorm=False, rmNan=False).squeeze()

        if 'testLstm' in doLst:
            df = app.streamflow.data.dbCsv.DataframeCsv(
                rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
            x = df.getData(
                varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
            model = train.model_load(outFolder, nEpoch, modelName='lstm_y1')
            yP = train.model_test(model, x).squeeze()
            ypLst.append(
                dbCsv.transNorm(yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
        if 'testCnn' in doLst:
            dfc = app.streamflow.data.dbCsv.DataframeCsv(
                rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
            xc = dfc.getData(
                varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
            yc = dfc.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
            yc[:, :, 0] = utils.interpNan(yc[:, :, 0])
            z = np.concatenate((yc, xc), axis=2)
            df = app.streamflow.data.dbCsv.DataframeCsv(
                rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
            x = df.getData(
                varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
            for opt in range(1, 4):
                modelName = 'cnn' + str(opt) + '_y1y1'
                model = train.model_load(outFolder, nEpoch, modelName=modelName)
                yP = train.model_test(model, x, z=z).squeeze()
                ypLst.append(
                    dbCsv.transNorm(
                        yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

        if 'post' in doLst:
            statDictLst = list()
            for k in range(0, len(ypLst)):
                statDictLst.append(post.statError(ypLst[k], yT))

            statStrLst = ['RMSE', 'ubRMSE', 'Bias', 'Corr']
            caseLst = ['LSTM', 'CNN-opt1', 'CNN-opt2', 'CNN-opt3']
            # caseLst = ['LSTM', 'CNN-opt2', 'CNN-opt3']
            postMat = np.ndarray([len(ypLst), len(statStrLst)])
            for iS in range(len(statStrLst)):
                statStr = statStrLst[iS]
                for k in range(len(ypLst)):
                    err = np.nanmean(statDictLst[k][statStr])
                    print('{} of {} = {:.5f}'.format(statStr, caseLst[k], err))
                    postMat[k, iS] = err
            np.set_printoptions(precision=4, suppress=True)
            print(postMat)

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
