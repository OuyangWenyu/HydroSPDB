import json
import os
import pickle as pkl
from collections import OrderedDict
import shutil

import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

from hydrospdb.data.source.data_base import DataSourceBase
from hydrospdb.data.source.data_camels import CAMELS_REGIONS
from hydrospdb.utils.hydro_stat import (
    cal_stat_basin_norm,
    cal_stat_prcp_norm,
    cal_stat_gamma,
    cal_stat,
    trans_norm,
)

SCALER_DICT = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
}


class ScalerHub(object):
    """
    A class for Scaler
    """

    def __init__(
        self,
        target_vars: np.array,
        relevant_vars: np.array,
        constant_vars: np.array = None,
        other_vars: dict = None,
        data_params: dict = None,
        loader_type: str = None,
        **kwargs
    ):
        """
        Perform normalization

        Parameters
        ----------
        target_vars
            output variables
        relevant_vars
            dynamic input variables
        constant_vars
            static input variables
        other_vars
            other required variables
        data_params
            parameters for reading data
        loader_type
            train, valid or test
        kwargs
            other optional parameters for ScalerHub
        """
        if not (
            target_vars.ndim == 3
            and relevant_vars.ndim == 3
            and constant_vars.ndim == 2
            and data_params is not None
            and loader_type is not None
        ):
            raise ValueError(
                "Please check if your read data correctly; the dimension of data is wrong"
            )
        scaler_type = data_params["scaler"]
        y_rm_nan = data_params["target_rm_nan"]
        x_rm_nan = data_params["relevant_rm_nan"]
        c_rm_nan = data_params["constant_rm_nan"]

        if scaler_type == "DapengScaler":
            assert "data_source" in list(kwargs.keys())
            gamma_norm_cols = data_params["scaler_params"]["gamma_norm_cols"]
            basin_norm_cols = data_params["scaler_params"]["basin_norm_cols"]
            if other_vars is not None:
                scaler = DapengScaler(
                    target_vars,
                    relevant_vars,
                    constant_vars,
                    data_params,
                    loader_type,
                    kwargs["data_source"],
                    other_vars=other_vars,
                    basin_norm_cols=basin_norm_cols,
                    gamma_norm_cols=gamma_norm_cols,
                )
                x, y, c, z = scaler.load_data(x_rm_nan, y_rm_nan, c_rm_nan)
            else:
                scaler = DapengScaler(
                    target_vars,
                    relevant_vars,
                    constant_vars,
                    data_params,
                    loader_type,
                    kwargs["data_source"],
                    basin_norm_cols=basin_norm_cols,
                    gamma_norm_cols=gamma_norm_cols,
                )
                x, y, c = scaler.load_data(x_rm_nan, y_rm_nan, c_rm_nan)
            self.target_scaler = scaler
        elif scaler_type in SCALER_DICT.keys():
            if other_vars is not None:
                raise Exception("not Implemented yet!!")
            all_vars = [target_vars, relevant_vars, constant_vars]
            norm_keys = ["target_vars", "relevant_vars", "constant_vars"]
            norm_dict = {}
            for i in range(len(all_vars)):
                data_tmp = all_vars[i]
                scaler = SCALER_DICT[scaler_type]()
                if data_tmp.ndim == 3:
                    # for forcings and outputs
                    num_instances, num_time_steps, num_features = data_tmp.shape
                    data_tmp = data_tmp.reshape(-1, num_features)
                    save_file = os.path.join(
                        data_params["test_path"], norm_keys[i] + "_scaler.pkl"
                    )
                    if loader_type == "train" and data_params["stat_dict_file"] is None:
                        data_norm = scaler.fit_transform(data_tmp)
                        # Save scaler in test_path for valid/test
                        with open(save_file, "wb") as outfile:
                            pkl.dump(scaler, outfile)
                    else:
                        if data_params["stat_dict_file"] is not None:
                            shutil.copy(data_params["stat_dict_file"], save_file)
                        if not os.path.isfile(save_file):
                            raise FileNotFoundError(
                                "Please genereate xx_scaler.pkl file"
                            )
                        with open(save_file, "rb") as infile:
                            scaler = pkl.load(infile)
                            data_norm = scaler.transform(data_tmp)
                    data_norm = data_norm.reshape(
                        num_instances, num_time_steps, num_features
                    )
                else:
                    # for attributes
                    save_file = os.path.join(
                        data_params["test_path"], norm_keys[i] + "_scaler.pkl"
                    )
                    if loader_type == "train" and data_params["stat_dict_file"] is None:
                        data_norm = scaler.fit_transform(data_tmp)
                        # Save scaler in test_path for valid/test
                        with open(save_file, "wb") as outfile:
                            pkl.dump(scaler, outfile)
                    else:
                        if data_params["stat_dict_file"] is not None:
                            shutil.copy(data_params["stat_dict_file"], save_file)
                        assert os.path.isfile(save_file)
                        with open(save_file, "rb") as infile:
                            scaler = pkl.load(infile)
                            data_norm = scaler.transform(data_tmp)
                norm_dict[norm_keys[i]] = data_norm
                if i == 0:
                    self.target_scaler = scaler
            x = norm_dict["relevant_vars"]
            if x_rm_nan:
                # As input, we cannot have NaN values
                x[np.where(np.isnan(x))] = 0
            y = norm_dict["target_vars"]
            if y_rm_nan:
                y[np.where(np.isnan(y))] = 0
            c = norm_dict["constant_vars"]
            if c_rm_nan:
                # As input, we cannot have NaN values
                c[np.where(np.isnan(c))] = 0
        else:
            raise NotImplementedError(
                "We don't provide this Scaler now!!! Please choose another one: DapengScaler or key in SCALER_DICT"
            )
        print("Finish Normalization\n")
        self.x = x
        self.y = y
        self.c = c
        if other_vars is not None:
            # TODO: no rm_nan for z
            self.z = z


class DapengScaler(object):
    def __init__(
        self,
        target_vars: np.array,
        relevant_vars: np.array,
        constant_vars: np.array,
        data_params: dict,
        loader_type: str,
        data_source: DataSourceBase,
        other_vars: dict = None,
        basin_norm_cols=[
            "usgsFlow",
            "streamflow",
            "ET",
            "ET_sum",
        ],
        gamma_norm_cols=[
            "prcp",
            "pr",
            "total_precipitation",
            "pet",
            "potential_evaporation",
            "PET",
        ],
    ):
        """
        The normalization and denormalization methods from Dapeng's 1st WRR paper.
        Some use StandardScaler, and some use special norm methods

        Parameters
        ----------
        target_vars
            output variables
        relevant_vars
            input dynamic variables
        constant_vars
            input static variables
        data_params
            data parameter config in data source
        loader_type
            train/valid/test
        data_source
            all config about data source
        other_vars
            if more input are needed, list them in other_vars
        basin_norm_cols
            data items which use _basin_norm method to normalize
        gamma_norm_cols
            data items which use log(\sqrt(x)+.1) method to normalize
        """
        camels_names = ["CAMELS_" + a_region for a_region in CAMELS_REGIONS]
        # now support "CAMELS", "CAMELS_DAYMET_V4", "NLDAS_CAMELS", "GAGES", and "GAGES_PRO" data_source
        if data_source.get_name() not in camels_names + [
            "CAMELS_DAYMET_V4",
            "ERA5LAND_CAMELS",
            "NLDAS_CAMELS",
            "CAMELS_FLOW_ET",
            "GAGES",
            "GAGES_PRO",
            "MODIS_ET_CAMELS",
            "CAMELS_SERIES",
        ]:
            raise NotImplementedError(
                "We cannot use this Scaler for "
                + data_source.get_name()
                + "\n Please Change to another Scaler"
            )
        self.data_target = target_vars
        self.data_forcing = relevant_vars
        self.data_attr = constant_vars
        self.data_source = data_source
        self.data_params = data_params
        self.t_s_dict = wrap_t_s_dict(data_source, data_params, loader_type)
        self.data_other = other_vars
        self.basin_norm_cols = basin_norm_cols
        self.gamma_norm_cols = gamma_norm_cols
        # both basin_norm_cols and gamma_norm_cols use log(\sqrt(x)+.1) method to normalize
        self.log_norm_cols = gamma_norm_cols + basin_norm_cols

        # save stat_dict of training period in test_path for valid/test
        stat_file = os.path.join(data_params["test_path"], "dapengscaler_stat.json")
        # for testing sometimes such as pub cases, we need stat_dict_file from trained dataset
        if loader_type == "train" and data_params["stat_dict_file"] is None:
            self.stat_dict = self.cal_stat_all()
            with open(stat_file, "w") as fp:
                json.dump(self.stat_dict, fp)
        else:
            # for valid/test, we need to load stat_dict from train
            if data_params["stat_dict_file"] is not None:
                # we used a assigned stat file, typically for PUB exps
                shutil.copy(data_params["stat_dict_file"], stat_file)
            assert os.path.isfile(stat_file)
            with open(stat_file, "r") as fp:
                self.stat_dict = json.load(fp)

    def inverse_transform(self, target_values):
        """
        Denormalization for output variables

        Parameters
        ----------
        target_values
            output variables

        Returns
        -------
        np.array
            denormalized predictions
        """
        stat_dict = self.stat_dict
        target_cols = self.data_params["target_cols"]
        pred = _trans_norm(
            target_values,
            target_cols,
            stat_dict,
            log_norm_cols=self.log_norm_cols,
            to_norm=False,
        )
        for i in range(len(self.data_params["target_cols"])):
            var = self.data_params["target_cols"][i]
            if var in self.basin_norm_cols:
                mean_prep = self.data_source.read_mean_prep(self.t_s_dict["sites_id"])
                if var in ["usgsFlow", "streamflow", "Q", "qobs"]:
                    # TODO: refactor unit of streamflow to mm/day
                    basin_area = self.data_source.read_basin_area(
                        self.t_s_dict["sites_id"]
                    )
                    pred[:, :, i : i + 1] = _basin_norm(
                        pred[:, :, i : i + 1], basin_area, mean_prep, to_norm=False
                    )
                else:
                    pred[:, :, i : i + 1] = _prcp_norm(
                        pred[:, :, i : i + 1], mean_prep, to_norm=False
                    )
        return pred

    def cal_stat_all(self):
        """
        Calculate statistics of outputs(streamflow etc), and inputs(forcing and attributes)

        Returns
        -------
        dict
            a dict with statistic values
        """
        # streamflow
        target_cols = self.data_params["target_cols"]
        stat_dict = dict()
        for i in range(len(target_cols)):
            var = target_cols[i]
            if var in self.basin_norm_cols:
                mean_prep = self.data_source.read_mean_prep(self.t_s_dict["sites_id"])
                if var in ["usgsFlow", "streamflow", "Q", "qobs"]:
                    # TODO: refactor unit of streamflow to mm/day,
                    # then we can remove these if statements with hard code,
                    # and use the same code for all basin_norm_cols variables
                    # only for streamflow with unit ft^3/s, transform it to m^3/day
                    flow = self.data_target[:, :, i]
                    basin_area = self.data_source.read_basin_area(
                        self.t_s_dict["sites_id"]
                    )
                    stat_dict[var] = cal_stat_basin_norm(flow, basin_area, mean_prep)
                else:
                    stat_dict[var] = cal_stat_prcp_norm(
                        self.data_target[:, :, i], mean_prep
                    )
            else:
                if var in self.gamma_norm_cols:
                    stat_dict[var] = cal_stat_gamma(self.data_target[:, :, i])
                else:
                    stat_dict[var] = cal_stat(self.data_target[:, :, i])

        # forcing
        # hard code for precipitation_str: prcp in daymet, pr in gridmet, total_precipitation in nldas
        forcing_lst = self.data_params["relevant_cols"]
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var in self.gamma_norm_cols:
                stat_dict[var] = cal_stat_gamma(x[:, :, k])
            else:
                stat_dict[var] = cal_stat(x[:, :, k])

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.data_params["constant_cols"]
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            stat_dict[var] = cal_stat(attr_data[:, k])

        # other vars
        if self.data_other is not None:
            other_data = self.data_other
            for key, value in other_data.items():
                stat_dict[key] = cal_stat(value)

        return stat_dict

    def get_data_obs(self, rm_nan: bool = False, to_norm: bool = True) -> np.array:
        """
        Get observation values

        Parameters
        ----------
        rm_nan
            if true, fill NaN value with 0
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the output value for modeling
        """
        stat_dict = self.stat_dict
        data = self.data_target
        target_cols = self.data_params["target_cols"]
        for i in range(len(target_cols)):
            var = target_cols[i]
            if var in self.basin_norm_cols:
                mean_prep = self.data_source.read_mean_prep(self.t_s_dict["sites_id"])
                if var in ["usgsFlow", "streamflow", "Q", "qobs"]:
                    # TODO: refactor unit of streamflow to mm/day
                    basin_area = self.data_source.read_basin_area(
                        self.t_s_dict["sites_id"]
                    )
                    data[:, :, i : i + 1] = _basin_norm(
                        data[:, :, i : i + 1], basin_area, mean_prep, to_norm=True
                    )
                else:
                    data[:, :, i : i + 1] = _prcp_norm(
                        data[:, :, i : i + 1], mean_prep, to_norm=True
                    )
        data = _trans_norm(
            data,
            target_cols,
            stat_dict,
            log_norm_cols=self.log_norm_cols,
            to_norm=to_norm,
        )
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, rm_nan=True, to_norm=True) -> np.array:
        """
        Get dynamic input data

        Parameters
        ----------
        rm_nan
            if true, fill NaN value with 0
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the dynamic inputs for modeling
        """
        stat_dict = self.stat_dict
        var_lst = self.data_params["relevant_cols"]
        data = self.data_forcing
        data = _trans_norm(
            data, var_lst, stat_dict, log_norm_cols=self.log_norm_cols, to_norm=to_norm
        )
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, rm_nan=True, to_norm=True) -> np.array:
        """
        Attr data and normalization

        Parameters
        ----------
        rm_nan
            if true, fill NaN value with 0
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the static inputs for modeling
        """
        stat_dict = self.stat_dict
        var_lst = self.data_params["constant_cols"]
        data = self.data_attr
        data = trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_other(self, rm_nan=True, to_norm=True) -> dict:
        """
        Other input data except for dynamic and static inputs

        Parameters
        ----------
        rm_nan
            if true, fill NaN value with 0
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            Other input data for modeling
        """
        stat_dict = self.stat_dict
        var_lst = list(self.data_params["other_cols"].keys())
        data = self.data_other
        trans_data = {}
        for var in var_lst:
            data_tmp = data[var]
            if data_tmp.ndim == 2:
                data_tmp = np.expand_dims(data_tmp, axis=2)
            data_tmp = trans_norm(data_tmp, var, stat_dict, to_norm=to_norm)
            if rm_nan is True:
                data_tmp[np.where(np.isnan(data_tmp))] = 0
            trans_data[var] = data_tmp
        return trans_data

    def load_data(self, x_rm_nan=True, y_rm_nan=False, c_rm_nan=True):
        """
        Read data and perform normalization for DL models

        Parameters
        ----------
        x_rm_nan
            if true, fill x's NaN value with 0
        y_rm_nan
            if true, fill y's NaN value with 0
        c_rm_nan
            if true, fill c's NaN value with 0

        Returns
        -------
        tuple
            x: 3-d  gages_num*time_num*var_num
            y: 3-d  gages_num*time_num*1
            c: 2-d  gages_num*var_num
        """
        x = self.get_data_ts(rm_nan=x_rm_nan)
        y = self.get_data_obs(rm_nan=y_rm_nan)
        c = self.get_data_const(rm_nan=c_rm_nan)
        if self.data_other is not None:
            z = self.get_data_other()
            return x, y, c, z
        return x, y, c


def _trans_norm(
    x: np.array, var_lst: list, stat_dict: dict, log_norm_cols: list, to_norm: bool
) -> np.array:
    """
    Normalization or inverse normalization

    There are two normalization formulas:

    .. math:: normalized_x = (x - mean) / std

    and

     .. math:: normalized_x = [log_{10}(\sqrt{x} + 0.1) - mean] / std

     The later is only for vars in log_norm_cols; mean is mean value; std means standard deviation

    Parameters
    ----------
    x
        data to be normalized or denormalized
    var_lst
        the type of variables
    stat_dict
        statistics of all variables
    log_norm_cols
        which cols use the second norm method
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.full(x.shape, np.nan)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                if var in log_norm_cols:
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var in log_norm_cols:
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var in log_norm_cols:
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var in log_norm_cols:
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
    return out


def _basin_norm(
    x: np.array, basin_area: np.array, mean_prep: np.array, to_norm: bool
) -> np.array:
    """
    Normalize or denormalize streamflow data with basin area and mean precipitation.

    The formula is as follows when normalizing (denormalize equation is its inversion):

    .. math:: normalized_x = \frac{x}{area * precipitation}

    Because units of streamflow, area, and precipitation are ft^3/s, km^2 and mm/day, respectively,
    and we need (m^3/day)/(m^3/day), we transform the equation as the code shows.

    Parameters
    ----------
    x
        data to be normalized or denormalized
    basin_area
        basins' area
    mean_prep
        basins' mean precipitation
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    nd = len(x.shape)
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basin_area, (1, x.shape[1]))
    tempprep = np.tile(mean_prep, (1, x.shape[1]))
    if to_norm is True:
        flow = (x * 0.0283168 * 3600 * 24) / (
            (temparea * (10**6)) * (tempprep * 10 ** (-3))
        )  # (m^3/day)/(m^3/day)
    else:
        flow = (
            x
            * ((temparea * (10**6)) * (tempprep * 10 ** (-3)))
            / (0.0283168 * 3600 * 24)
        )
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow


def _prcp_norm(x: np.array, mean_prep: np.array, to_norm: bool) -> np.array:
    """
    Normalize or denormalize data with mean precipitation.

    The formula is as follows when normalizing (denormalize equation is its inversion):

    .. math:: normalized_x = \frac{x}{precipitation}

    Parameters
    ----------
    x
        data to be normalized or denormalized
    mean_prep
        basins' mean precipitation
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    nd = len(x.shape)
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    tempprep = np.tile(mean_prep, (1, x.shape[1]))
    if to_norm:
        flow = x / tempprep  # (mm/day)/(mm/day)
    else:
        flow = x * tempprep
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow


def wrap_t_s_dict(
    data_source: DataSourceBase, data_params: dict, loader_type: str
) -> OrderedDict:
    """
    Basins and periods

    Parameters
    ----------
    data_source
        source data object
    data_params
        Parameters for reading from data source
    loader_type
        train, valid or test

    Returns
    -------
    OrderedDict
        OrderedDict(sites_id=basins_id, t_final_range=t_range_list)
    """
    basins_id = data_params["object_ids"]
    if type(basins_id) is str and basins_id == "ALL":
        basins_id = data_source.read_object_ids().tolist()
    # assert all(x < y for x, y in zip(basins_id, basins_id[1:]))
    if "t_range_" + loader_type in data_params:
        t_range_list = data_params["t_range_" + loader_type]
    else:
        raise Exception(
            "Error! The mode "
            + str(loader_type)
            + " was not found in the data_source params dict. Please add it."
        )
    t_s_dict = OrderedDict(sites_id=basins_id, t_final_range=t_range_list)
    return t_s_dict
