"""target：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
   procedure： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
from data import *
from explore.stat import stat_ind
from hydroDL.master import *
from utils import send_email
from visual import *

print('loading package hydroDL')
# TODO：多GPU计算
config_file = "../../data/config.ini"

# 读取模型配置文件
optTrain, optModel, optLoss = init_model_param(config_file)
model_dict = wrap_master(optModel, optLoss, optTrain)

# 准备训练数据
source_data = SourceData(config_file, optTrain.get("tRangeTrain"))

# 构建输入数据类对象
data_model = DataModel(source_data)

# 进行模型训练
# train model
master_train(data_model, model_dict)
# 训练结束，发送email，email中给一个输出文件夹的提示
out = source_data.all_configs['out']
send_email.send_email(subject='Training Done', text=out)

# test
# 首先构建test时的data和model，然后调用test函数计算
source_data_test = SourceData(config_file, optTrain.get("tRangeTest"))
test_data_model = DataModel(source_data_test)
df, pred, obs = master_test(test_data_model, model_dict)

# 统计性能指标
inds = stat_ind(obs, pred)

# plot box，使用seaborn库
plot_box_inds(inds)

# plot time series
plot_ts_obs_pred(obs, pred)
