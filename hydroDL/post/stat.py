import numpy as np
import scipy.stats
from hydroeval import evaluator, nse

keyLst = ['Bias', 'RMSE', 'ubRMSE', 'Corr', 'NSE']


def statError(pred, target):
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target) ** 2, axis=1))
    # ubRMSE
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom) ** 2, axis=1))
    # rho
    Corr = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            Corr[k] = scipy.stats.pearsonr(xx, yy)[0]

    # nse
    nse_ = np.array([evaluator(nse, pred[i], target[i]) for i in range(pred.shape[0])]).squeeze()
    outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, NSE=nse_)
    return outDict


def statError1d(pred, target):
    # Bias
    Bias = np.mean(pred - target)
    # RMSE
    RMSE = np.sqrt(np.mean((pred - target) ** 2))
    # nse
    nse_ = evaluator(nse, pred, target)[0]

    return [Bias, RMSE, nse_]


def cal_stat(x):
    a = x.flatten()
    print(~np.isnan(a))
    b = a[~np.isnan(a)]
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def trans_norm(x, var_lst, stat_dict, *, to_norm):
    """归一化计算方法，包括反向的计算过程，测试的时候需要还原数据"""
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.zeros(x.shape)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
    return out


if __name__ == "__main__":
    print("main")
    arrays = {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9], "b": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    print(arrays)
    a_percentile_keys = [key for key in arrays.keys()]
    a_percentile_values = [np.percentile(value, [0, 25, 50, 75, 100]) for value in arrays.values()]
    a_percentile = {a_percentile_keys[k]: a_percentile_values[k] for k in range(len(a_percentile_keys))}
    print(a_percentile)

    b_percentile = {key: np.percentile(value, [0, 25, 50, 75, 100]) for key, value in arrays.items()}
    print(b_percentile)

    # 取一个数组小于30%分位数的所有值
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = a[np.where(a < np.percentile(a, 30))]
    print(b)

    a = np.array([2, 6, 3, 8, 4, 1, 5, 7, 9])
    print(np.where(a < np.percentile(a, 30)))
    b = a[np.where(a < np.percentile(a, 30))]
    print(b)

    c = np.array([[[3], [1]], [[2], [4]]])
    print(c.squeeze().shape)
    print(c.shape)
    print(c)
    print(c.reshape(c.shape[0], c.shape[1]).shape)

    a = np.array([[2, 6, 3, 8, 4, 1, 5, 7, 9], [2, 3, 4, np.nan, 6, 7, 8, 9, 10]])
    print(a.shape)
    a[np.where(np.isnan(a))] = 0
    b = a[np.where(a < np.percentile(a, 30))]
    print(b)

    a_index = [np.where(a_i < np.percentile(a_i, 30)) for a_i in a]
    print(a_index)
    b = [a[i][a_index[i]].tolist() for i in range(a.shape[0])]
    print(type(b))
    c = np.array(b)
    print(c.shape)

    print(statError1d(np.array([1, 2, 3]), np.array([4, 5, 6])))
