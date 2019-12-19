"""target：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
   procedure： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
from data.data_source import SourceData
from data.read_config import init_path, init_data_param, init_model_param
from data.data_process import wrap_master, namePred, basic_statistic

# 读取配置文件# 首先，配置GAGES-II原始数据文件的路径。在hydroDL的init文件里直接配置输入输出路径
import os
from explore.stat import stat_ind
from hydroDL.master import run_train, test
from visual.plot import plot_box_fig, plot_ts

print('loading package hydroDL')

config_file = "../../data/config.ini"
pathDataSource = init_path(config_file)

# 读取配置文件
optData = init_data_param(config_file)
optModel, optLoss, optTrain, optTest = init_model_param(config_file)

cid = 0
gpuNum = 3
case = optData['subset'] + '-' + str(optData['tRange'][0])[2:4] + '-' + str(optData['tRange'][1])[2:4]
out = os.path.join(pathDataSource['Out'], case)
masterDict = wrap_master(out, optData, optModel, optLoss, optTrain)

# 准备数据并读取
source_data = SourceData(config_file, optData.get("tRange"))

# 初步统计计算
basic_statistic()

# 计算完成统计值进行模型训练
# train model
# see whether there are previous results or not, if yes, there is no need to train again.
caseLst = [case]
outLst = [os.path.join(pathDataSource['Out'], x) for x in caseLst]
# 从测试配置项中获取相应参数
tRangeTest, subset = optTest

resultPathLst = namePred(out, tRangeTest, subset, epoch=optTrain['nEpoch'])
if not os.path.exists(resultPathLst[0]):
    run_train(masterDict, cuda_id=cid % gpuNum, screen='test')
    cid = cid + 1

# test
predLst = list()
for out in outLst:
    df, pred, obs = test(out, t_range=tRangeTest, subset=subset, epoch=optTrain['nEpoch'])
    predLst.append(pred)

# 统计性能指标
inds = stat_ind()

# plot box，使用seaborn库
plot_box_fig()

# plot time series
plot_ts()
