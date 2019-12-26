"""target：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
   procedure： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
from data import *
from explore.stat import stat_ind
from hydroDL.master import *
from utils import hydro_util
from visual import *

print('starting hydroDL...')
# TODO：多GPU计算
configFile = r"../../data/config.ini"

# 读取模型配置文件
optTrain, optData, optModel, optLoss = init_model_param(configFile)
modelDict = wrap_master(optData, optModel, optLoss, optTrain)

# 准备训练数据
sourceData = SourceData(configFile, optData.get("tRangeTrain"), ['1980-01-01', '2015-01-01'])

# 构建输入数据类对象
dataModel = DataModel(sourceData)

# 进行模型训练
# train model
master_train(dataModel, modelDict)
# 训练结束，发送email，email中给一个输出文件夹的提示
out = sourceData.all_configs['out']
hydro_util.send_email(subject='Training Done', text=out)

# test
# 首先构建test时的data和model，然后调用test函数计算
sourceDataTest = SourceData(configFile, optTrain.get("tRangeTest"))
testDataModel = DataModel(sourceDataTest)
df, pred, obs = master_test(testDataModel, modelDict)

# 统计性能指标
inds = stat_ind(obs, pred)

# plot box，使用seaborn库
plot_box_inds(inds)

# plot time series
plot_ts_obs_pred(obs, pred)
