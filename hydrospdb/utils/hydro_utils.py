"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2022-12-19 19:52:03
LastEditors: Wenyu Ouyang
Description: Util functions
FilePath: /HydroSPB/hydroSPB/utils/hydro_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import json
import os
import re
import zipfile
import datetime as dt, datetime
from typing import Union
import geopandas as gpd
import pickle
import smtplib
import ssl
from collections import OrderedDict
import numpy as np
import pandas as pd
import urllib
from urllib import parse

import requests
import matplotlib.pyplot as plt
import torch
from itertools import combinations

import threading
import functools

import tqdm

import logging

# -----------------------------------------------logger setting----------------------------------------------------
from torch_scatter import segment_csr, scatter


def get_hydro_logger(log_level_param):
    logger = logging.getLogger(__name__)
    # StreamHandler
    stream_handler = logging.StreamHandler()  # console stream output
    stream_handler.setLevel(level=log_level_param)
    logger.addHandler(stream_handler)
    return logger


log_level = logging.INFO
hydro_logger = get_hydro_logger(log_level)


# ------------------------------------------------progress bar----------------------------------------------------
def provide_progress_bar(
    function, estimated_time, tstep=0.2, tqdm_kwargs={}, args=[], kwargs={}
):
    """
    Tqdm wrapper for a long-running function

    Parameters
    -----------
    function
        function to run
    estimated_time
        how long you expect the function to take
    tstep
        time delta (seconds) for progress bar updates
    tqdm_kwargs
        kwargs to construct the progress bar
    args
        args to pass to the function
    kwargs
        keyword args to pass to the function

    Returns
    --------
    function(*args, **kwargs)
    """
    ret = [None]  # Mutable var so the function can store its return value

    def myrunner(function, ret, *args, **kwargs):
        ret[0] = function(*args, **kwargs)

    thread = threading.Thread(
        target=myrunner, args=(function, ret) + tuple(args), kwargs=kwargs
    )
    pbar = tqdm.tqdm(total=estimated_time, **tqdm_kwargs)

    thread.start()
    while thread.is_alive():
        thread.join(timeout=tstep)
        pbar.update(tstep)
    pbar.close()
    return ret[0]


def progress_wrapped(estimated_time, tstep=0.2, tqdm_kwargs={}):
    """Decorate a function to add a progress bar"""

    def real_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return provide_progress_bar(
                function,
                estimated_time=estimated_time,
                tstep=tstep,
                tqdm_kwargs=tqdm_kwargs,
                args=args,
                kwargs=kwargs,
            )

        return wrapper

    return real_decorator


def setup_log(tag="VOC_TOPICS"):
    # create logger
    logger = logging.getLogger(tag)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)
    return logger


# -------------------------------------------------- notification tools--------------------------------------------
def send_email(subject, text, receiver="hust2014owen@gmail.com"):
    sender = "hydro.wyouyang@gmail.com"
    password = "D4VEFya3UQxGR3z"
    context = ssl.create_default_context()
    msg = "Subject: {}\n\n{}".format(subject, text)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(from_addr=sender, to_addrs=receiver, msg=msg)


# -------------------------------------------------plot utils-------------------------------------------------------
def save_or_show_plot(file_nm: str, save: bool, save_path=""):
    if save:
        plt.savefig(os.path.join(save_path, file_nm))
    else:
        plt.show()


# ----------------------------------------------------data tools--------------------------------------------------------
def unzip_file(data_zip, path_unzip):
    """extract a zip file"""
    with zipfile.ZipFile(data_zip, "r") as zip_temp:
        zip_temp.extractall(path_unzip)


def unzip_nested_zip(dataset_zip, path_unzip):
    """
    Extract a zip file including any nested zip files

    If a file's name is "xxx_", it seems the "extractall" function in the "zipfile" lib will throw an OSError,
    so please check the unzipped files manually when this occurs.

    Parameters
    ----------
    dataset_zip: the zip file
    path_unzip: where it is unzipped
    """

    with zipfile.ZipFile(dataset_zip, "r") as zfile:
        try:
            zfile.extractall(path=path_unzip)
        except OSError as e:
            logging.warning(
                "Please check the unzipped files manually. There may be some missed important files."
            )
            logging.warning("The directory is: " + path_unzip)
    for root, dirs, files in os.walk(path_unzip):
        for filename in files:
            if re.search(r"\.zip$", filename):
                file_spec = os.path.join(root, filename)
                new_dir = os.path.join(root, filename[0:-4])
                unzip_nested_zip(file_spec, new_dir)


def zip_file_name_from_url(data_url, data_dir):
    data_url_str = data_url.split("/")
    filename = parse.unquote(data_url_str[-1])
    zipfile_path = os.path.join(data_dir, filename)
    unzip_dir = os.path.join(data_dir, filename[0:-4])
    return zipfile_path, unzip_dir


def is_there_file(zipfile_path, unzip_dir):
    """if a file has existed"""
    if os.path.isfile(zipfile_path):
        if os.path.isdir(unzip_dir):
            return True
        unzip_nested_zip(zipfile_path, unzip_dir)
        return True


def download_one_zip(data_url, data_dir):
    """
    download one zip file from url as data_file

    Parameters
    ----------
    data_url: the URL of the downloading website
    data_dir: where we will put the data
    """

    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.makedirs(unzip_dir)
        r = requests.get(data_url, stream=True)
        with open(zipfile_path, "wb") as py_file:
            for chunk in r.iter_content(chunk_size=1024):  # 1024 bytes
                if chunk:
                    py_file.write(chunk)
        unzip_nested_zip(zipfile_path, unzip_dir), download_small_file


def download_small_zip(data_url, data_dir):
    """download zip file and unzip"""
    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.mkdir(unzip_dir)
        zipfile_path, _ = urllib.request.urlretrieve(data_url, zipfile_path)
        unzip_nested_zip(zipfile_path, unzip_dir)


def download_small_file(data_url, temp_file):
    """download data from url to the temp_file"""
    r = requests.get(data_url)
    with open(temp_file, "w") as f:
        f.write(r.text)


def download_excel(data_url, temp_file):
    """download a excel file according to url"""
    if not os.path.isfile(temp_file):
        urllib.request.urlretrieve(data_url, temp_file)


def download_a_file_from_google_drive(drive, dir_id, download_dir):
    file_list = drive.ListFile(
        {"q": "'" + dir_id + "' in parents and trashed=false"}
    ).GetList()
    for file in file_list:
        print("title: %s, id: %s" % (file["title"], file["id"]))
        file_dl = drive.CreateFile({"id": file["id"]})
        print("mimetype is %s" % file_dl["mimeType"])
        if file_dl["mimeType"] == "application/vnd.google-apps.folder":
            download_dir_sub = os.path.join(download_dir, file_dl["title"])
            if not os.path.isdir(download_dir_sub):
                os.makedirs(download_dir_sub)
            download_a_file_from_google_drive(drive, file_dl["id"], download_dir_sub)
        else:
            # download
            temp_file = os.path.join(download_dir, file_dl["title"])
            if os.path.isfile(temp_file):
                print("file has been downloaded")
                continue
            file_dl.GetContentFile(os.path.join(download_dir, file_dl["title"]))
            print("Downloading file finished")


def serialize_json(my_dict, my_file):
    with open(my_file, "w") as FP:
        json.dump(my_dict, FP, indent=4)


def unserialize_json_ordered(my_file):
    # m_file = os.path.join(my_file, 'master.json')
    with open(my_file, "r") as fp:
        m_dict = json.load(fp, object_pairs_hook=OrderedDict)
    return m_dict


def unserialize_json(my_file):
    with open(my_file, "r") as fp:
        my_object = json.load(fp)
    return my_object


def serialize_pickle(my_object, my_file):
    f = open(my_file, "wb")
    pickle.dump(my_object, f)
    f.close()


def unserialize_pickle(my_file):
    f = open(my_file, "rb")
    my_object = pickle.load(f)
    f.close()
    return my_object


def serialize_numpy(my_array, my_file):
    np.save(my_file, my_array)


def unserialize_numpy(my_file):
    y = np.load(my_file)
    return y


def serialize_geopandas(gpd_df, my_file, the_driver="GeoJSON"):
    gpd_df.to_file(my_file, driver=the_driver)


def unserialize_geopandas(my_file):
    gpd_df = gpd.read_file(my_file)
    return gpd_df


def check_np_array_nan(func):
    """
    Check if the return numpy array will include NaN. If true, raise a warning

    Parameters
    ----------
    func
        function to run

    Returns
    -------
    function(*args, **kwargs)
        the wrapper
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if type(result) in [tuple, list]:
            count = 0
            for an_array in result:
                if type(an_array) is dict:
                    for key in an_array:
                        if np.isnan(an_array[key]).any():
                            hydro_logger.warning(
                                "Please check your input data: there are NaN data! It may affect following calculation!!\n "
                                "The location of NaN values in the "
                                + str(count)
                                + "-th dict are:\n"
                            )
                            hydro_logger.warning("value of " + key + ":\n")
                            hydro_logger.warning(np.argwhere(np.isnan(an_array[key])))
                else:
                    if np.isnan(an_array).any():
                        hydro_logger.warning(
                            "Please check your input data: there are NaN data! It may affect following calculation!!\n "
                            "The location of NaN values in the "
                            + str(count)
                            + "-th array are:\n"
                        )
                        hydro_logger.warning(np.argwhere(np.isnan(an_array)))
                count = count + 1
        elif type(result) is np.array:
            if np.isnan(result).any():
                hydro_logger.warning(
                    "Please check your input data: there are NaN data! It may affect following calculation!!\n "
                    "The location of NaN values are:\n"
                )
                hydro_logger.warning(np.argwhere(np.isnan(result)))
        return result

    return wrapper


# -------------------------------------------------time & date tools--------------------------------------------------
def t2str(t_: Union[str, dt.datetime]):
    if type(t_) is str:
        t_str = dt.datetime.strptime(t_, "%Y-%m-%d")
        return t_str
    elif type(t_) is dt.datetime:
        t = t_.strftime("%Y-%m-%d")
        return t
    else:
        raise NotImplementedError("We don't support this data type yet")


def t_range_days(t_range, *, step=np.timedelta64(1, "D")) -> np.array:
    """
    Transform the two-value t_range list to a uniformly-spaced list (default is a daily list).
    For example, ["2000-01-01", "2000-01-05"] -> ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]

    Parameters
    ----------
    t_range
        two-value t_range list
    step
        the time interval; its default value is 1 day

    Returns
    -------
    np.array
        a uniformly-spaced (daily) list
    """
    sd = dt.datetime.strptime(t_range[0], "%Y-%m-%d")
    ed = dt.datetime.strptime(t_range[1], "%Y-%m-%d")
    return np.arange(sd, ed, step)


def t_range_days_inv(t_days_lst):
    """A inverse function of t_range_days

    Parameters
    ----------
    t_days_lst : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    t_train_lst = pd.date_range(
        start=np.datetime64(t_days_lst[0]),
        end=np.datetime64(t_days_lst[-1]) + np.timedelta64(1, "D"),
        freq="D",
    )
    return t_days_lst2range(t_train_lst)


def t_range_days_timedelta(t_array, td=12, td_type="h"):
    """
    for each day, add a timedelta

    Parameters
    ----------
    t_array
        its data type is same as the return type of "t_range_days" function
    td
        time periods
    td_type
        the type of time period

    Returns
    -------
    np.array
        a new t_array
    """
    assert td_type in ["Y", "M", "D", "h", "m", "s"]
    t_array_final = [t + np.timedelta64(td, td_type) for t in t_array]
    return np.array(t_array_final)


def t_days_lst2range(t_array: list) -> list:
    """
    Transform a period list to its interval.
    For example,  ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"] ->  ["2000-01-01", "2000-01-04"]

    Parameters
    ----------
    t_array: list[Union[np.datetime64, str]]
        a period list

    Returns
    -------
    list
        An time interval
    """
    if type(t_array[0]) == np.datetime64:
        t0 = t_array[0].astype(datetime.datetime)
        t1 = t_array[-1].astype(datetime.datetime)
    else:
        t0 = t_array[0]
        t1 = t_array[-1]
    sd = t0.strftime("%Y-%m-%d")
    ed = t1.strftime("%Y-%m-%d")
    return [sd, ed]


def t_range_years(t_range):
    """t_range is a left-closed and right-open interval, if t_range[1] is not Jan.1 then end_year should be included"""
    start_year = int(t_range[0].split("-")[0])
    end_year = int(t_range[1].split("-")[0])
    end_month = int(t_range[1].split("-")[1])
    end_day = int(t_range[1].split("-")[2])
    if end_month == 1 and end_day == 1:
        year_range_list = np.arange(start_year, end_year)
    else:
        year_range_list = np.arange(start_year, end_year + 1)
    return year_range_list


def get_year(a_time):
    if isinstance(a_time, datetime.date):
        return a_time.year
    elif isinstance(a_time, np.datetime64):
        return a_time.astype("datetime64[Y]").astype(int) + 1970
    else:
        return int(a_time[:4])


def intersect(t_lst1, t_lst2):
    C, ind1, ind2 = np.intersect1d(t_lst1, t_lst2, return_indices=True)
    return ind1, ind2


def date_to_julian(a_time):
    if type(a_time) == str:
        fmt = "%Y-%m-%d"
        dt = datetime.datetime.strptime(a_time, fmt)
    else:
        dt = a_time
    tt = dt.timetuple()
    julian_date = tt.tm_yday
    return julian_date


def t_range_to_julian(t_range):
    t_array = t_range_days(t_range)
    t_array_str = np.datetime_as_string(t_array)
    julian_dates = [date_to_julian(a_time[0:10]) for a_time in t_array_str]
    return julian_dates


# --------------------------------------------------MATH CALCULATION---------------------------------------------------
def subset_of_dict(dict, chosen_keys):
    """make a new dict from key-values of chosen keys in a list"""
    return {key: value for key, value in dict.items() if key in chosen_keys}


def pair_comb(combine_attrs):
    if len(combine_attrs) == 1:
        values = list(combine_attrs[0].values())[0]
        key = list(combine_attrs[0].keys())[0]
        results = []
        for value in values:
            d = dict()
            d[key] = value
            results.append(d)
        return results
    items_all = list()
    for dict_item in combine_attrs:
        list_temp = list(dict_item.values())[0]
        items_all = items_all + list_temp
    all_combs = list(combinations(items_all, 2))

    def is_in_same_item(item1, item2):
        for dict_item in combine_attrs:
            list_now = list(dict_item.values())[0]
            if item1 in list_now and item2 in list_now:
                return True
        return False

    def which_dict(item):
        for dict_item in combine_attrs:
            list_now = list(dict_item.values())[0]
            if item in list_now:
                return list(dict_item.keys())[0]

    combs = [comb for comb in all_combs if not is_in_same_item(comb[0], comb[1])]
    combs_dict = [
        {which_dict(comb[0]): comb[0], which_dict(comb[1]): comb[1]} for comb in combs
    ]
    return combs_dict


def flat_data(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return xSort


def interpNan(x, mode="linear"):
    if len(x.shape) == 1:
        ngrid = 1
        nt = x.shape[0]
    else:
        ngrid, nt = x.shape
    for k in range(ngrid):
        xx = x[k, :]
        xx = interpNan1d(xx, mode)
    return x


def interpNan1d(x, mode="linear"):
    i0 = np.where(np.isnan(x))[0]
    i1 = np.where(~np.isnan(x))[0]
    if len(i1) > 0:
        if mode == "linear":
            x[i0] = np.interp(i0, i1, x[i1])
        if mode == "pre":
            x0 = x[i1[0]]
            for k in range(len(x)):
                if k in i0:
                    x[k] = x0
                else:
                    x0 = x[k]

    return x


def is_any_elem_in_a_lst(lst1, lst2, return_index=False, include=False):
    do_exist = False
    idx_lst = []
    for j in range(len(lst1)):
        if include:
            for lst2_elem in lst2:
                if lst1[j] in lst2_elem:
                    idx_lst.append(j)
                    do_exist = True
        else:
            if lst1[j] in lst2:
                idx_lst.append(j)
                do_exist = True
    if return_index:
        return do_exist, idx_lst
    return do_exist


def random_choice_no_return(arr, num_lst):
    """sampling without replacement multi-times, and the num of each time is in num_lst"""
    num_lst_arr = np.array(num_lst)
    num_sum = num_lst_arr.sum()
    if type(arr) == list:
        arr = np.array(arr)
    assert num_sum <= arr.size
    results = []
    arr_residue = np.arange(arr.size)
    for num in num_lst_arr:
        idx_chosen = np.random.choice(arr_residue.size, num, replace=False)
        chosen_idx_in_arr = np.sort(arr_residue[idx_chosen])
        results.append(arr[chosen_idx_in_arr])
        arr_residue = np.delete(arr_residue, idx_chosen)
    return results


def find_integer_factors_close_to_square_root(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


def nanlog(x):
    if x != x:
        return np.nan
    else:
        return np.log(x)


def random_index(
    ngrid: int, nt: int, dim_subset: Union[tuple, list], warmup_length=0
) -> tuple:
    """
    A similar function with PyTorch's Dataset's Sampler function -- random index

    Parameters
    ----------
    ngrid
        number of all basins/grids
    nt
        number of all periods
    dim_subset
        [batch_size, rho]
    warmup_length
        if warmup_length>0, it means we need some time-steps' calculation before model's formal forward.
        This variable is manly for Physics-based models, which always need warmup to get proper initial state variables

    Returns
    -------
    tuple[np.array, np.array]
        indices of grids and periods
    """
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, [batch_size])
    i_t = np.random.randint(0 + warmup_length, nt - rho, [batch_size])
    return i_grid, i_t


def select_subset(
    x: np.array,
    i_grid: np.array,
    i_t: np.array,
    rho: int,
    warmup_length: int = 0,
    c: np.array = None,
    tuple_out: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Select one mini-batch seq-first tensor data ([seq, batch, feature]) from dataset

    Parameters
    ----------
    x
        source data [batch, seq, feature]
    i_grid
        i-th basin/grid/...; from "random_index" function
    i_t
        i-th period; from "random_index" function
        when i_t is None, it means that we will only use attribute data from c
    rho
        the length of time series
    warmup_length
        the length of warmup periods
    c
        static attribute data
    tuple_out
        if True, return (x,c); otherwise return concatenated x-c

    Returns
    -------
    torch.Tensor
        a mini-batch seq-first tensor; [seq, batch, feature]
        or a mini-batch two-dim tensor: [batch, feature]; this case is only for inputs with only attributes data
    """
    if i_t is None:
        # No time sequence data, just for static attribute data
        if c is not None and c.shape[-1] > 0:
            c_tensor = torch.from_numpy(c[i_grid, :]).float()
            return c_tensor
        # when c is None and i_t is also None, x is special, such as a sequence used for kernel
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(i_grid):  # hack
        i_grid = np.arange(0, len(i_grid))  # hack
        if nt <= rho + warmup_length:
            i_t.fill(0)
    if i_t is not None:
        batch_size = i_grid.shape[0]
        x_tensor = torch.zeros(
            [rho + warmup_length, batch_size, nx], requires_grad=False
        )
        for k in range(batch_size):
            temp = x[
                i_grid[k] : i_grid[k] + 1,
                np.arange(i_t[k] - warmup_length, i_t[k] + rho),
                :,
            ]
            x_tensor[:, k : k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if len(x.shape) == 2:
            # keep sequence first!
            x_tensor = torch.from_numpy(np.swapaxes(x[i_grid, :], 1, 0)).float()
        else:
            x_tensor = torch.from_numpy(np.swapaxes(x[i_grid, :, :], 1, 0)).float()
            rho = x_tensor.shape[0]
    if c is not None and c.shape[-1] > 0:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho + warmup_length, axis=1
        )
        c_tensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()
        if tuple_out:
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                c_tensor = c_tensor.cuda()
            out = (x_tensor, c_tensor)
        else:
            out = torch.cat((x_tensor, c_tensor), 2)
    else:
        out = x_tensor
    return out


def select_subset_batch_first(
    x, i_grid, i_t, rho, warmup_length=0, c=None
) -> torch.Tensor:
    """
    Select one mini-batch batch-first tensor data ([batch, seq, feature]) from dataset

    Parameters
    ----------
    x
        source data [batch, seq, feature]
    i_grid
        i-th basin/grid/...; from "random_index" function
    i_t
        i-th period; from "random_index" function
        when i_t is None, it means that we will only use attribute data from c
    rho
        the length of time series
    warmup_length
        the length of warmup periods
    c
        static attribute data

    Returns
    -------
    torch.Tensor
        a mini-batch batch-first tensor; [batch, seq, feature]
        or a mini-batch two-dim tensor: [batch, feature]; this case is only for inputs with only attributes data
    """
    if i_t is None:
        # No time sequence data, just for static attribute data
        if c is None or c.shape[-1] == 0:
            raise ArithmeticError(
                "Need attribute data but find None! Please check your input!!"
            )
        else:
            c_tensor = torch.from_numpy(c[i_grid, :]).float()
            return c_tensor
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] < len(i_grid):
        raise ValueError("grid num should be smaller than x.shape[0]")
    if nt < rho + warmup_length:
        raise ValueError("time length option should be larger than rho")

    batch_size = i_grid.shape[0]
    x_tensor = torch.zeros([batch_size, rho + warmup_length, nx], requires_grad=False)
    for k in range(batch_size):
        x_tensor[k : k + 1, :, :] = torch.from_numpy(
            x[
                i_grid[k] : i_grid[k] + 1,
                np.arange(i_t[k] - warmup_length, i_t[k] + rho),
                :,
            ]
        ).float()

    if c is not None and c.shape[-1] > 0:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho + warmup_length, axis=1
        )
        c_tensor = torch.from_numpy(temp).float()
        out = torch.cat((x_tensor, c_tensor), 2)
    else:
        out = x_tensor
    return out


def deal_gap_data(output, target, data_gap, device):
    """
    How to handle with gap data

    When there are NaN values in observation, we will perform a "reduce" operation on prediction.
    For example, pred = [0,1,2,3,4], obs=[5, nan, nan, 6, nan]; the "reduce" is sum;
    then, pred_sum = [0+1+2, 3+4], obs_sum=[5,6], loss = loss_func(pred_sum, obs_sum).
    Notice: when "sum", actually final index is not chosen,
    because the whole observation may be [5, nan, nan, 6, nan, nan, 7, nan, nan], 6 means sum of three elements.
    Just as the rho is 5, the final one is not chosen

    Parameters
    ----------
    output
        model output for k-th variable
    target
        target for k-th variable
    data_gap
        data_gap=1： reduce is sum
        data_gap=2: reduce is mean
    device
        where to save the data

    Returns
    -------
    tuple[tensor, tensor]
        output and target after dealing with gap
    """
    # all members in a batch has different NaN-gap, so we need a loop
    seg_p_lst = []
    seg_t_lst = []
    for j in range(target.shape[1]):
        non_nan_idx = [
            i for i in range(len(target[:, j])) if not torch.isnan(target[i, j])
        ]
        scatter_index = []
        idx_tmp = 0
        if len(non_nan_idx) < 1:
            raise ArithmeticError("All NaN elements, please check your data")
        for i in range(non_nan_idx[0], len(target[:, j])):
            if i > non_nan_idx[0] and (not torch.isnan(target[i, j])):
                idx_tmp = idx_tmp + 1
            scatter_index.append(idx_tmp)
        if data_gap == 1:
            seg = segment_csr(
                output[:, j],
                torch.tensor(non_nan_idx).to(device=device),
                reduce="sum",
            )
        elif data_gap == 2:
            if not non_nan_idx[-1] == len(target[:, j]):
                # this will be used in t_j = target[non_nan_idx[:-1], j, k]
                non_nan_idx = non_nan_idx + [len(target[:, j])]
            seg = scatter(
                output[non_nan_idx[0] :, j],
                torch.tensor(scatter_index).to(device=device),
                reduce="mean",
            )
            # the following code cause grad nan, so we use scatter rather than segment_csr.
            # But notice start index of output become non_nan_idx[0] rather than 0
            # seg = segment_csr(output[:, j], torch.tensor(non_nan_idx).to(device=self.device),
            #                   reduce="mean")
        else:
            raise NotImplementedError(
                "We have not provided this reduce way now!! Please choose 1 or 2!!"
            )
        seg_p_lst.append(seg)
        # t0_j = target[:, j, k]
        # mask_j = t0_j == t0_j
        # t_j = t0_j[mask_j]
        # Accordingly, we only chose target[non_nan_idx[:-1], j, k] rather than [non_nan_idx, j, k]
        t_j = target[non_nan_idx[:-1], j]
        seg_t_lst.append(t_j)
    p = torch.cat(seg_p_lst)
    t = torch.cat(seg_t_lst)
    return p, t
