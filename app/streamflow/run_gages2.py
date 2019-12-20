"""target：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
   procedure： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
from data.data_source import SourceData
from data.read_config import init_path, init_data_param, init_model_param, wrap_master, namePred
from data.data_input import DataModel

from explore.stat import stat_ind
from hydroDL.master import train, test
from utils import send_email
from visual.plot import plot_box_fig, plot_ts

print('loading package hydroDL')
# TODO：多GPU计算
config_file = "../../data/config.ini"

# 读取模型配置文件
optModel, optLoss, optTrain, optTest = init_model_param(config_file)
model_dict = wrap_master(optModel, optLoss, optTrain, optTest)

# 准备训练数据
source_data = SourceData(config_file, optTrain.get("tRangeTrain"))

# 构建输入数据类对象
data_model = DataModel(source_data)

# 计算完成统计值进行模型训练
# train model
train(data_model, model_dict)
# email中给一个输出文件夹的提示
out = source_data.all_configs['out']
send_email.sendEmail(subject='Training Done', text=out)

# test
df, pred, obs = test(out, t_range=tRangeTest, subset=subset, epoch=optTrain['nEpoch'])

# 统计性能指标
inds = stat_ind()

# plot box，使用seaborn库
plot_box_fig()

# plot time series
plot_ts()
