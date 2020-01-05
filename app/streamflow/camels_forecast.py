"""
target：utilizing camels/gages dataset to train LSTM model and test
pipeline:
    data pre-processing——statistical analysis——model training and testing——visualization of outputs——tune parameters

目的：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
基本流程： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
from data import *
from explore.stat import statError
from hydroDL.master import *
from utils import hydro_util
from visual import *
import numpy as np

print('Starting ...')
# TODO：多GPU计算
configFile = r"../../data/config.ini"

# 读取模型配置文件
configData = CamelsConfig(configFile)

# 准备训练数据
sourceData = CamelsSource(configData, configData.model_dict["data"]["tRangeTrain"])

# 构建输入数据类对象
dataModel = DataModel(sourceData)

# 进行模型训练
# train model
master_train(dataModel)
# 训练结束，发送email，email中给一个输出文件夹的提示
hydro_util.send_email(subject='Training Done', text=configData.model_dict['dir']['Out'])

# test
# 首先构建test时的data和model，然后调用test函数计算
sourceDataTest = CamelsSource(configData, configData.model_dict["data"]["tRangeTest"])
testDataModel = DataModel(sourceDataTest)
df, pred, obs = master_test(testDataModel)

# 统计性能指标
pred = pred.reshape(pred.shape[0], pred.shape[1])
obs = obs.reshape(pred.shape[0], pred.shape[1])
inds = statError(obs, pred)
# plot box，使用seaborn库
plot_box_inds(inds)
# plot time series
show_me_num = 5
t_s_dict = testDataModel.t_s_dict
sites = np.array(t_s_dict["sites_id"])
t_range = np.array(t_s_dict["t_final_range"])
plot_ts_obs_pred(obs, pred, sites, t_range, show_me_num)
