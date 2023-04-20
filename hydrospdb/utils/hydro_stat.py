"""
Author: MHPI group, Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-04-20 17:51:07
LastEditors: Wenyu Ouyang
Description: statistics calculation
FilePath: /HydroSPDB/hydrospdb/utils/hydro_stat.py
Copyright (c) 2021-2022 MHPI group, Wenyu Ouyang. All rights reserved.
"""

import copy
import itertools
import HydroErr as he
import numpy as np
import scipy.stats
from scipy.stats import wilcoxon
import pandas as pd
from hydrospdb.utils.hydro_utils import hydro_logger

ALL_METRICS = ["Bias", "RMSE", "ubRMSE", "Corr", "R2", "NSE", "KGE", "FHV", "FLV"]


def fms(obs, sim, lower: float = 0.2, upper: float = 0.7) -> float:
    r"""
    TODO: not fully tested
    Calculate the slope of the middle section of the flow duration curve [#]_

    .. math::
        \%\text{BiasFMS} = \frac{\left | \log(Q_{s,\text{lower}}) - \log(Q_{s,\text{upper}}) \right | -
            \left | \log(Q_{o,\text{lower}}) - \log(Q_{o,\text{upper}}) \right |}{\left |
            \log(Q_{s,\text{lower}}) - \log(Q_{s,\text{upper}}) \right |} \times 100,

    where :math:`Q_{s,\text{lower/upper}}` corresponds to the FDC of the simulations (here, `sim`) at the `lower` and
    `upper` bound of the middle section and :math:`Q_{o,\text{lower/upper}}` similarly for the observations (here,
    `obs`).

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    lower : float, optional
        Lower bound of the middle section in range ]0,1[, by default 0.2
    upper : float, optional
        Upper bound of the middle section in range ]0,1[, by default 0.7

    Returns
    -------
    float
        Slope of the middle section of the flow duration curve.

    References
    ----------
    .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process-based diagnostic approach to model
        evaluation: Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417,
        doi:10.1029/2007WR006716.
    """
    if len(obs) < 1:
        return np.nan

    if any([(x <= 0) or (x >= 1) for x in [upper, lower]]):
        raise ValueError("upper and lower have to be in range ]0,1[")

    if lower >= upper:
        raise ValueError("The lower threshold has to be smaller than the upper.")

    # get arrays of sorted (descending) discharges
    obs = np.sort(obs)
    sim = np.sort(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    # calculate fms part by part
    qsm_lower = np.log(sim[np.round(lower * len(sim)).astype(int)])
    qsm_upper = np.log(sim[np.round(upper * len(sim)).astype(int)])
    qom_lower = np.log(obs[np.round(lower * len(obs)).astype(int)])
    qom_upper = np.log(obs[np.round(upper * len(obs)).astype(int)])

    fms = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / (
        qom_lower - qom_upper + 1e-6
    )

    return fms * 100


def mean_peak_timing(
    obs, sim, window: int = None, resolution: str = "1D", datetime_coord: str = None
) -> float:
    """
    TODO: not finished
    Mean difference in peak flow timing.

    Uses scipy.find_peaks to find peaks in the observed time series. Starting with all observed peaks, those with a
    prominence of less than the standard deviation of the observed time series are discarded. Next, the lowest peaks
    are subsequently discarded until all remaining peaks have a distance of at least 100 steps. Finally, the
    corresponding peaks in the simulated time series are searched in a window of size `window` on either side of the
    observed peaks and the absolute time differences between observed and simulated peaks is calculated.
    The final metric is the mean absolute time difference across all peaks. For more details, see Appendix of [#]_

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    window : int, optional
        Size of window to consider on each side of the observed peak for finding the simulated peak. That is, the total
        window length to find the peak in the simulations is :math:`2 * \\text{window} + 1` centered at the observed
        peak. The default depends on the temporal resolution, e.g. for a resolution of '1D', a window of 3 is used and
        for a resolution of '1H' the the window size is 12.
    resolution : str, optional
        Temporal resolution of the time series in pandas format, e.g. '1D' for daily and '1H' for hourly.
    datetime_coord : str, optional
        Name of datetime coordinate. Tried to infer automatically if not specified.


    Returns
    -------
    float
        Mean peak time difference.

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple
        meteorological datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci.,
        https://doi.org/10.5194/hess-2020-221
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations (scipy's find_peaks doesn't guarantee correctness with NaNs)
    obs, sim = _mask_valid(obs, sim)

    # heuristic to get indices of peaks and their corresponding height.
    peaks, _ = signal.find_peaks(
        obs.values, distance=100, prominence=np.std(obs.values)
    )

    # infer name of datetime index
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(obs)

    if window is None:
        # infer a reasonable window size
        window = max(int(utils.get_frequency_factor("12H", resolution)), 3)

    # evaluate timing
    timing_errors = []
    for idx in peaks:
        # skip peaks at the start and end of the sequence and peaks around missing observations
        # (NaNs that were removed in obs & sim would result in windows that span too much time).
        if (
            (idx - window < 0)
            or (idx + window >= len(obs))
            or (
                pd.date_range(
                    obs[idx - window][datetime_coord].values,
                    obs[idx + window][datetime_coord].values,
                    freq=resolution,
                ).size
                != 2 * window + 1
            )
        ):
            continue

        # check if the value at idx is a peak (both neighbors must be smaller)
        if (sim[idx] > sim[idx - 1]) and (sim[idx] > sim[idx + 1]):
            peak_sim = sim[idx]
        else:
            # define peak around idx as the max value inside of the window
            values = sim[idx - window : idx + window + 1]
            peak_sim = values[values.argmax()]

        # get xarray object of qobs peak, for getting the date and calculating the datetime offset
        peak_obs = obs[idx]

        # calculate the time difference between the peaks
        delta = peak_obs.coords[datetime_coord] - peak_sim.coords[datetime_coord]

        timing_error = np.abs(delta.values / pd.to_timedelta(resolution))

        timing_errors.append(timing_error)

    return np.mean(timing_errors) if len(timing_errors) > 0 else np.nan


def KGE(xs, xo):
    """
    Kling Gupta Efficiency (Gupta et al., 2009, http://dx.doi.org/10.1016/j.jhydrol.2009.08.003)
    input:
        xs: simulated
        xo: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    r = np.corrcoef(xo, xs)[0, 1]
    alpha = np.std(xs) / np.std(xo)
    beta = np.mean(xs) / np.mean(xo)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def stat_error_i(targ_i, pred_i):
    """statistics for one"""
    ind = np.where(np.logical_and(~np.isnan(pred_i), ~np.isnan(targ_i)))[0]
    # Theoretically at least two points for correlation
    if ind.shape[0] > 1:
        xx = pred_i[ind]
        yy = targ_i[ind]
        bias = he.me(xx, yy)
        # RMSE
        rmse = he.rmse(xx, yy)
        # ubRMSE
        pred_mean = np.nanmean(xx)
        target_mean = np.nanmean(yy)
        pred_anom = xx - pred_mean
        target_anom = yy - target_mean
        ubrmse = np.sqrt(np.nanmean((pred_anom - target_anom) ** 2))
        # rho R2 NSE
        corr = he.pearson_r(xx, yy)
        r2 = he.r_squared(xx, yy)
        nse = he.nse(xx, yy)
        kge = he.kge_2009(xx, yy)
        # percent bias
        pbias = np.sum(xx - yy) / np.sum(yy) * 100
        # FHV the peak flows bias 2%
        # FLV the low flows bias bottom 30%, log space
        pred_sort = np.sort(xx)
        target_sort = np.sort(yy)
        indexlow = round(0.3 * len(pred_sort))
        indexhigh = round(0.98 * len(pred_sort))
        lowpred = pred_sort[:indexlow]
        highpred = pred_sort[indexhigh:]
        lowtarget = target_sort[:indexlow]
        hightarget = target_sort[indexhigh:]
        pbiaslow = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
        pbiashigh = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
        out_dict = dict(
            Bias=bias,
            RMSE=rmse,
            ubRMSE=ubrmse,
            Corr=corr,
            R2=r2,
            NSE=nse,
            KGE=kge,
            FHV=pbiashigh,
            FLV=pbiaslow,
        )
        return out_dict
    else:
        raise ValueError(
            "The number of data is less than 2, we don't calculate the statistics."
        )


def stat_error(target: np.array, pred: np.array, fill_nan: str = "no") -> dict:
    """
    Statistics indicators include: Bias, RMSE, ubRMSE, Corr, R2, NSE, KGE, FHV, FLV

    Parameters
    ----------
    target
        observations, typically 2-dim, when it is 3-dim, set a loop for final dim
    pred
        predictions
    fill_nan
        "no" means ignoring the NaN value, and it is the default setting;
        "sum" means calculate the sum of the following values in the NaN locations.
        For example, observations are [1, nan, nan, 2], and predictions are [0.3, 0.3, 0.3, 1.5].
        Then, "no" means [1, 2] v.s. [0.3, 1.5] while "sum" means [1, 2] v.s. [0.3 + 0.3 + 0.3, 1.5];
        "mean" represents calculate average value the following values in the NaN locations.

    Returns
    -------
    dict
        Bias, RMSE, ubRMSE, Corr, R2, NSE, KGE, FHV, FLV
    """
    if len(target.shape) == 3:
        assert type(fill_nan) in [list, tuple, np.ndarray]
        dict_list = []
        if type(fill_nan) is not list or len(fill_nan) != target.shape[-1]:
            raise RuntimeError("Please give more fill_nan choices")
        for k in range(target.shape[-1]):
            k_dict = stat_error(target[:, :, k], pred[:, :, k], fill_nan=fill_nan[k])
            dict_list.append(k_dict)
        return dict_list
    if len(target.shape) == 2:
        if type(fill_nan) is list or type(fill_nan) is tuple:
            fill_nan = fill_nan[0]
    assert type(fill_nan) is str
    if fill_nan != "no":
        each_non_nan_idx = []
        all_non_nan_idx = []
        for i in range(target.shape[0]):
            tmp = target[i]
            non_nan_idx_tmp = [j for j in range(tmp.size) if not np.isnan(tmp[j])]
            each_non_nan_idx.append(non_nan_idx_tmp)
            # TODO: now all_non_nan_idx is only set for ET, because of its irregular nan values
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp
            non_nan_idx = np.unique(all_non_nan_idx).tolist()
        # some NaN data appear in different dates in different basins, so we have to calculate the metric for each basin
        # but for ET, it is not very resonable to calculate the metric for each basin in this way, for example,
        # the non_nan_idx: [1, 9, 17, 33, 41], then there are 16 elements in 17 -> 33, so use all_non_nan_idx is better
        # hence we don't use each_non_nan_idx finally
        out_dict = dict(
            Bias=[],
            RMSE=[],
            ubRMSE=[],
            Corr=[],
            R2=[],
            NSE=[],
            KGE=[],
            FHV=[],
            FLV=[],
        )
    if fill_nan == "sum":
        for i in range(target.shape[0]):
            tmp = target[i]
            # non_nan_idx = each_non_nan_idx[i]
            targ_i = tmp[non_nan_idx]
            pred_i = np.add.reduceat(pred[i], non_nan_idx)
            dict_i = stat_error_i(targ_i, pred_i)
            out_dict["Bias"].append(dict_i["Bias"])
            out_dict["RMSE"].append(dict_i["RMSE"])
            out_dict["ubRMSE"].append(dict_i["ubRMSE"])
            out_dict["Corr"].append(dict_i["Corr"])
            out_dict["R2"].append(dict_i["R2"])
            out_dict["NSE"].append(dict_i["NSE"])
            out_dict["KGE"].append(dict_i["KGE"])
            out_dict["FHV"].append(dict_i["FHV"])
            out_dict["FLV"].append(dict_i["FLV"])
        return out_dict
    elif fill_nan == "mean":
        hydro_logger.debug(
            "calculate mean value in an interval between two non-nan value"
        )
        for i in range(target.shape[0]):
            tmp = target[i]
            # non_nan_idx = each_non_nan_idx[i]
            targ_i = tmp[non_nan_idx]
            pred_i_sum = np.add.reduceat(pred[i], non_nan_idx)
            if non_nan_idx[-1] < len(pred[i]):
                idx4mean = non_nan_idx + [len(pred[i])]
            else:
                idx4mean = copy.copy(non_nan_idx)
            idx_interval = [y - x for x, y in zip(idx4mean, idx4mean[1:])]
            pred_i = pred_i_sum / idx_interval
            dict_i = stat_error_i(targ_i, pred_i)
            out_dict["Bias"].append(dict_i["Bias"])
            out_dict["RMSE"].append(dict_i["RMSE"])
            out_dict["ubRMSE"].append(dict_i["ubRMSE"])
            out_dict["Corr"].append(dict_i["Corr"])
            out_dict["R2"].append(dict_i["R2"])
            out_dict["NSE"].append(dict_i["NSE"])
            out_dict["KGE"].append(dict_i["KGE"])
            out_dict["FHV"].append(dict_i["FHV"])
            out_dict["FLV"].append(dict_i["FLV"])
        return out_dict
    # TODO: refactor Dapeng's code
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
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    KGe = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    num_lowtarget_zero = 0
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100
            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                yymean = yy.mean()
                SST = np.sum((yy - yymean) ** 2)
                SSReg = np.sum((xx - yymean) ** 2)
                SSRes = np.sum((yy - xx) ** 2)
                R2[k] = 1 - SSRes / SST
                NSE[k] = 1 - SSRes / SST
                KGe[k] = KGE(xx, yy)
            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.98 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            if np.sum(lowtarget) == 0:
                num_lowtarget_zero = num_lowtarget_zero + 1
            PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
            PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
    outDict = dict(
        Bias=Bias,
        RMSE=RMSE,
        ubRMSE=ubRMSE,
        Corr=Corr,
        R2=R2,
        NSE=NSE,
        KGE=KGe,
        FHV=PBiashigh,
        FLV=PBiaslow,
    )
    hydro_logger.debug(
        "The CDF of BFLV will not reach 1.0 because some basins have all zero flow observations for the "
        "30% low flow interval, the percent bias can be infinite\n"
        + "The number of these cases is "
        + str(num_lowtarget_zero)
    )
    return outDict


def cal_4_stat_inds(b):
    """
    Calculate four statistics indices: percentile 10 and 90, mean value, standard deviation

    Parameters
    ----------
    b
        input data

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def cal_stat(x: np.array) -> list:
    """
    Get statistic values of x (Exclude the NaN values)

    Parameters
    ----------
    x: the array

    Returns
    -------
    list
        [10% quantile, 90% quantile, mean, std]
    """
    a = x.flatten()
    b = a[~np.isnan(a)]
    if b.size == 0:
        # if b is [], then give it a 0 value
        b = np.array([0])
    return cal_4_stat_inds(b)


def cal_stat_gamma(x):
    """
    Try to transform a time series data to normal distribution

    Now only for daily streamflow, precipitation and evapotranspiration;
    When nan values exist, just ignore them.

    Parameters
    ----------
    x
        time series data

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    return cal_4_stat_inds(b)


def cal_stat_basin_norm(x, basinarea, meanprep):
    """
    normalized daily streamflow by basin area and precipitation with cal_stat_gamma

    Parameters
    ----------
    x
    basinarea
        basinarea = readAttr(gageDict['id'], ['area_gages2'])
    meanprep
        meanprep = readAttr(gageDict['id'], ['p_mean'])

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    temparea = np.tile(basinarea, (1, x.shape[1]))
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    flowua = (x * 0.0283168 * 3600 * 24) / (
        (temparea * (10**6)) * (tempprep * 10 ** (-3))
    )  # unit (m^3/day)/(m^3/day)
    return cal_stat_gamma(flowua)


def cal_stat_prcp_norm(x, meanprep):
    """
    normalized daily evapotranspiration or soil moisture by precipitation with cal_stat_gamma

    Parameters
    ----------
    x
        data to be normalized
    meanprep
        meanprep = readAttr(gageDict['id'], ['p_mean'])

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    # unit (mm/day)/(mm/day)
    flowua = x / tempprep
    return cal_stat_gamma(flowua)


def trans_norm(x, var_lst, stat_dict, *, to_norm):
    """
    normalization, including denormalization code

    Parameters
    ----------
    x
        2d or 3d data
        2d：1st-sites，2nd-var type
        3d：1st-sites，2nd-time, 3rd-var type
    var_lst
        variables
    stat_dict
        a dict with statistics info
    to_norm
        if True, normalization; else denormalization

    Returns
    -------
    np.array
        normalized/denormalized data
    """
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


def ecdf(data):
    """Compute ECDF"""
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return (x, y)


def wilcoxon_t_test(xs, xo):
    """Wilcoxon t test"""
    diff = xs - xo  # same result when using xo-xs
    w, p = wilcoxon(diff)
    return w, p


def wilcoxon_t_test_for_lst(x_lst, rnd_num=2):
    """Wilcoxon t test for every two array in a 2-d array"""
    arr_lst = np.asarray(x_lst)
    w, p = [], []
    arr_lst_pair = list(itertools.combinations(arr_lst, 2))
    for arr_pair in arr_lst_pair:
        wi, pi = wilcoxon_t_test(arr_pair[0], arr_pair[1])
        w.append(round(wi, rnd_num))
        p.append(round(pi, rnd_num))
    return w, p


def cal_fdc(data: np.array, quantile_num=100):
    # data = n_grid * n_day
    n_grid, n_day = data.shape
    fdc = np.full([n_grid, quantile_num], np.nan)
    for ii in range(n_grid):
        temp_data0 = data[ii, :]
        temp_data = temp_data0[~np.isnan(temp_data0)]
        # deal with no data case for some gages
        if len(temp_data) == 0:
            temp_data = np.full(n_day, 0)
        # sort from large to small
        temp_sort = np.sort(temp_data)[::-1]
        # select quantile_num quantile points
        n_len = len(temp_data)
        ind = (np.arange(quantile_num) / quantile_num * n_len).astype(int)
        fdc_flow = temp_sort[ind]
        if len(fdc_flow) != quantile_num:
            raise Exception("unknown assimilation variable")
        else:
            fdc[ii, :] = fdc_flow

    return fdc

def remove_abnormal_data(data, *, q1=0.00001, q2=0.99999):
    """
    remove abnormal data

    Parameters
    ----------
    data
        data to be removed
    q
        lower quantile
    q2
        upper quantile

    Returns
    -------
    np.array
        data after removing abnormal data
    """
    # remove abnormal data
    data[data < np.quantile(data, q1)] = np.nan
    data[data > np.quantile(data, q2)] = np.nan
    return data

def month_stat_for_daily_df(df):
    """
    calculate monthly statistics for daily data

    Parameters
    ----------
    df
        daily data

    Returns
    -------
    pd.DataFrame
        monthly statistics for daily data
    """
    # guarantee the index is datetime
    df.index = pd.to_datetime(df.index)
    return df.resample('MS').mean()
