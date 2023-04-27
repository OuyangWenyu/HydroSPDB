"""
Author: Wenyu Ouyang
Date: 2022-01-08 17:31:35
LastEditTime: 2023-04-27 16:41:17
LastEditors: Wenyu Ouyang
Description: Some util functions for scripts in app/streamflow
FilePath: /HydroSPDB/scripts/streamflow_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import csv
import shutil
from functools import reduce
import os
import sys
from pathlib import Path
import fnmatch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tbparse import SummaryReader


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydrospdb.utils.hydro_utils import unserialize_json, random_choice_no_return
from hydrospdb.models.trainer import (
    load_result,
    save_result,
    train_and_evaluate,
)
from hydrospdb.utils.hydro_stat import ecdf
from hydrospdb.data.config import default_config_file, update_cfg, cmd
from hydrospdb.data.source.data_gages import Gages
from hydrospdb.visual.plot_stat import plot_ecdfs_matplot
from scripts.script_constant import VAR_C_CHOSEN_FROM_GAGES_II, VAR_T_CHOSEN_FROM_DAYMET


def get_json_file(cfg_dir):
    json_files_lst = []
    json_files_ctime = []
    for file in os.listdir(cfg_dir):
        if (
            fnmatch.fnmatch(file, "*.json")
            and "_stat" not in file  # statistics json file
            and "_dict" not in file  # data cache json file
        ):
            json_files_lst.append(os.path.join(cfg_dir, file))
            json_files_ctime.append(os.path.getctime(os.path.join(cfg_dir, file)))
    sort_idx = np.argsort(json_files_ctime)
    cfg_file = json_files_lst[sort_idx[-1]]
    cfg_json = unserialize_json(cfg_file)
    if cfg_json["data_params"]["test_path"] != cfg_dir:
        # sometimes we will use files copied from other device, so the dir is not correct for this device
        update_cfg_as_move_to_another_pc(cfg_json=cfg_json)

    return cfg_json


def update_cfg_as_move_to_another_pc(cfg_json):
    """update cfg as move to another pc

    Returns
    -------
    _type_
        _description_
    """
    cfg_json["data_params"]["test_path"] = get_the_new_path_with_diff_part(
        cfg_json, "test_path"
    )
    cfg_json["data_params"]["data_path"] = get_the_new_path_with_diff_part(
        cfg_json, "data_path"
    )
    cfg_json["data_params"]["cache_path"] = get_the_new_path_with_diff_part(
        cfg_json, "cache_path"
    )
    cfg_json["data_params"]["validation_path"] = get_the_new_path_with_diff_part(
        cfg_json, "validation_path"
    )


def get_the_new_path_with_diff_part(cfg_json, replace_item):
    the_item = cfg_json["data_params"][replace_item]
    if the_item is None:
        return None
    common_path_name = get_common_path_name(the_item, replace_item)
    dff_path_name = (
        definitions.DATASET_DIR if replace_item == "data_path" else definitions.ROOT_DIR
    )
    if type(common_path_name) is list:
        return [os.path.join(dff_path_name, a_path) for a_path in common_path_name]
    return os.path.join(dff_path_name, common_path_name)


def get_common_path_name(origin_pc_test_path, replace_item):
    if type(origin_pc_test_path) is list:
        return [
            get_common_path_name(a_path, replace_item) for a_path in origin_pc_test_path
        ]
    if origin_pc_test_path.startswith("/"):
        # linux
        origin_pc_test_path_lst = origin_pc_test_path.split("/")
    else:
        # windows
        origin_pc_test_path_lst = origin_pc_test_path.split("\\")
    # NOTE: this is a hard code
    if replace_item == "data_path":
        pos_lst = [i for i, e in enumerate(origin_pc_test_path_lst) if e == "data"]
        the_root_dir = definitions.DATASET_DIR
    else:
        pos_lst = [i for i, e in enumerate(origin_pc_test_path_lst) if e == "HydroSPB"]
        the_root_dir = definitions.ROOT_DIR
    if not pos_lst:
        raise ValueError("Can not find the common path name")
    elif len(pos_lst) == 1:
        where_start_same_in_origin_pc = pos_lst[0] + 1
    else:
        for i in pos_lst:
            if os.path.exists(
                os.path.join(
                    the_root_dir, os.sep.join(origin_pc_test_path_lst[i + 1 :])
                )
            ):
                where_start_same_in_origin_pc = i + 1
                break

    return os.sep.join(origin_pc_test_path_lst[where_start_same_in_origin_pc:])


def get_latest_file_in_a_lst(lst):
    """get the latest file in a list

    Parameters
    ----------
    lst : list
        list of files

    Returns
    -------
    str
        the latest file
    """
    lst_ctime = [os.path.getctime(file) for file in lst]
    sort_idx = np.argsort(lst_ctime)
    return lst[sort_idx[-1]]


def get_lastest_weight_path(weight_path_dir):
    """Get the last modified weight file

    Parameters
    ----------
    weight_path_dir : _type_
        _description_

    Returns
    -------
    str
        the path of the weight file
    """
    pth_files_lst = [
        os.path.join(weight_path_dir, file)
        for file in os.listdir(weight_path_dir)
        if fnmatch.fnmatch(file, "*.pth")
    ]
    return get_latest_file_in_a_lst(pth_files_lst)


def evaluate_a_model(
    exp,
    example="camels",
    epoch=None,
    train_period=None,
    test_period=None,
    save_result_name=None,
    is_tl=False,
    sub_exp=None,
    data_dir=None,
    device=None,
    dpl_param=None,
):
    """
    Evaluate a trained model

    Parameters
    ----------
    exp
        the name of exp, such as "exp511"
    example
        first sub-dir in "example" directory: "camels", "gages", ... default is the former
    epoch
        model saved in which epoch is used here
    train_period
        the period of training data for model
    test_period
        the period of testing data for model
    save_result_name
        the name of the result file, default is None
    sub_exp
        the name of sub exp, default is None,
        which is for saved models during training for different hyper-parameters settings
    data_dir
        the directory of data source, default is None
        when move the trained model from one machine to another, this will be useful

    Returns
    -------
    None
    """
    cfg_dir_flow = os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", example, exp
    )
    cfg_flow = get_json_file(cfg_dir_flow)
    if data_dir is not None:
        cfg_flow["data_params"]["data_path"] = data_dir
        cfg_flow["data_params"]["test_path"] = cfg_dir_flow
        cfg_flow["data_params"]["cache_path"] = cfg_dir_flow
        cfg_flow["data_params"]["validation_path"] = cfg_dir_flow
    cfg_flow["model_params"]["continue_train"] = False
    if train_period is not None:
        cfg_flow["data_params"]["t_range_train"] = train_period
        cfg_flow["data_params"]["cache_read"] = False
        # don't save the cache file,
        # because we will evaluate the model with different settings in the same directory
        cfg_flow["data_params"]["cache_write"] = False
    if test_period is not None:
        cfg_flow["data_params"]["t_range_test"] = test_period
        cfg_flow["data_params"]["cache_read"] = False
        cfg_flow["data_params"]["cache_write"] = False
    if epoch is None:
        epoch = cfg_flow["evaluate_params"]["test_epoch"]
    else:
        # the epoch is used for test
        cfg_flow["evaluate_params"]["test_epoch"] = epoch
    train_epoch = cfg_flow["training_params"]["epochs"]
    if epoch != train_epoch:
        cfg_flow["training_params"]["epochs"] = epoch
    if device is not None:
        cfg_flow["training_params"]["device"] = device
    if sub_exp is not None:
        weight_path = os.path.join(cfg_dir_flow, sub_exp, f"model_Ep{str(epoch)}.pth")
    else:
        weight_path = os.path.join(cfg_dir_flow, f"model_Ep{str(epoch)}.pth")
    if not os.path.isfile(weight_path):
        weight_path = os.path.join(cfg_dir_flow, f"model_Ep{str(epoch)}.pt")
    cfg_flow["model_params"]["weight_path"] = weight_path
    if cfg_flow["data_params"]["cache_read"]:
        cfg_flow["data_params"]["cache_write"] = False
    if is_tl:
        # we evaluate a tl model, so we need to set tl_tag to False to avoid it perform tl modeling again
        cfg_flow["model_params"]["model_param"]["tl_tag"] = False
    if dpl_param is not None:
        cfg_flow["model_params"]["model_param"].update(dpl_param)
    train_and_evaluate(cfg_flow)
    # new name for results, becuase we will evaluate the model with different settings in the same directory
    pred, obs = load_result(
        cfg_flow["data_params"]["test_path"], epoch, not_only_1out=True
    )
    if save_result_name is None:
        pred_name = "flow_pred"
        obs_name = "flow_obs"
    else:
        pred_name = save_result_name + "_pred"
        obs_name = save_result_name + "_obs"
    save_result(
        cfg_flow["data_params"]["test_path"],
        epoch,
        pred,
        obs,
        pred_name=pred_name,
        obs_name=obs_name,
    )
    print("Call a trained model and save its evaluation results")


def predict_in_test_period_with_model(new_exp_args, cache_cfg_dir, weight_path):
    """Prediction in a test period with the given trained model for a new experiment

    Parameters
    ----------
    new_exp_args : str
        arguments for new experiment
    cache_cfg_dir : str
        the directory of cache file
    weight_path : str
        the path of trained model's weight file
    """
    cfg = default_config_file()
    update_cfg(cfg, new_exp_args)
    if cache_cfg_dir is not None:
        # test_data_dict.json is a flag for cache existing. 
        # If exists, we don't need to write cache file again
        cfg["data_params"]["cache_write"] = not os.path.exists(
            os.path.join(cache_cfg_dir, "test_data_dict.json")
        )
        cfg["data_params"]["cache_read"] = True
    cfg["model_params"]["continue_train"] = False
    cfg["model_params"]["weight_path"] = weight_path
    if weight_path is None:
        cfg["model_params"]["continue_train"] = True
    train_and_evaluate(cfg)
    print("Call a trained model and test it in a new period")


def plot_ecdf_func(
    inds_all_lst,
    cases_exps_legends_together,
    save_path,
    dash_lines=None,
    ecdf_fig_size=(6, 4),
    colors="rbkgcmy",
    x_str="NSE",
    x_interval=0.1,
    x_lim=(0, 1),
    show_legend=True,
    legend_font_size=16,
):
    """
    Plot ECDF figs for a list of NSE arrays

    Parameters
    ----------
    inds_all_lst
        list of NSE arrays
    cases_exps_legends_together
        exps' names for the legend
    save_path
        where we save the fig
    dash_lines
        if the line will be a dash line
    ecdf_fig_size
        fig's size
    colors
        colors for each line
    x_str
        the name of x-axis
    x_interval
        interval of x-axis
    x_lim
        limits of x-axis
    show_legend
        if True, show legend

    Returns
    -------
    plt.figure
        the plot
    """
    if dash_lines is None:
        dash_lines = [True, False, False, False, False, False]
    print("plot CDF")
    xs1 = []
    ys1 = []
    for ind_ in inds_all_lst:
        xi, yi = ecdf(np.nan_to_num(ind_))
        xs1.append(xi)
        ys1.append(yi)
    fig, ax = plot_ecdfs_matplot(
        xs1,
        ys1,
        cases_exps_legends_together,
        dash_lines=dash_lines,
        x_str=x_str,
        y_str="CDF",
        colors=colors,
        fig_size=ecdf_fig_size,
        x_interval=x_interval,
        x_lim=x_lim,
        show_legend=show_legend,
        legend_font_size=legend_font_size,
    )
    plt.tight_layout()
    FIGURE_DPI = 600
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def predict_new_gages_exp(
    exp,
    gage_id_file,
    weight_path,
    continue_train,
    train_period,
    test_period,
    cache_path=None,
    gages_id=None,
    stat_dict_file=None,
    var_c=VAR_C_CHOSEN_FROM_GAGES_II,
    var_t=VAR_T_CHOSEN_FROM_DAYMET,
):
    project_name = "gages/" + exp
    args = cmd(
        sub=project_name,
        source_path=definitions.DATASET_DIR,
        source="GAGES",
        download=0,
        ctx=[0],
        model_name="KuaiLSTM",
        model_param={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        weight_path=weight_path,
        continue_train=continue_train,
        opt="Adadelta",
        loss_func="RMSESum",
        rs=1234,
        # train_period is just used for a dummy value, not used in the prediction
        train_period=train_period,
        test_period=test_period,
        cache_write=1,
        cache_read=1,
        scaler="DapengScaler",
        data_loader="StreamflowDataModel",
        train_epoch=300,
        te=300,
        save_epoch=20,
        batch_size=100,
        rho=365,
        var_c=var_c,
        var_t=var_t,
        gage_id_file=gage_id_file,
        gage_id=gages_id,
        stat_dict_file=stat_dict_file,
    )
    predict_in_test_period_with_model(
        args, weight_path=weight_path, cache_cfg_dir=cache_path
    )


def split_samples_for_ecoregions(
    selected_ids: list, split_num, save=False, kfold_dir=None
):
    """split samples to training and test, but both shoud have basins from each ecoregion

    Parameters
    ----------
    selected_ids : list
        the basins' ids
    split_num:
        the k of k-fold cross validation
    save : bool
        if save, training and test basins ids will be saved as csv files
    kfold_dir: str
        the saved directory, if save

    Returns
    -------
    tuple
        (list(training basins ids), list(testing basins ids))
    """
    random_seed = 1234
    eco_names = [
        ("ECO2_CODE", 5.2),
        ("ECO2_CODE", 5.3),
        ("ECO2_CODE", 6.2),
        ("ECO2_CODE", 7.1),
        ("ECO2_CODE", 8.1),
        ("ECO2_CODE", 8.2),
        ("ECO2_CODE", 8.3),
        ("ECO2_CODE", 8.4),
        ("ECO2_CODE", 8.5),
        ("ECO2_CODE", 9.2),
        ("ECO2_CODE", 9.3),
        ("ECO2_CODE", 9.4),
        ("ECO2_CODE", 9.5),
        ("ECO2_CODE", 9.6),
        ("ECO2_CODE", 10.1),
        ("ECO2_CODE", 10.2),
        ("ECO2_CODE", 10.4),
        ("ECO2_CODE", 11.1),
        ("ECO2_CODE", 12.1),
        ("ECO2_CODE", 13.1),
    ]
    np.random.seed(random_seed)
    kf = KFold(n_splits=split_num, shuffle=True, random_state=random_seed)
    # eco attr exists in Gages
    gages_data_path = os.path.join(definitions.DATASET_DIR, "gages")
    gages = Gages(gages_data_path)
    eco_name_chosen = []
    sites_lst_train = []
    sites_lst_test = []
    for eco_name in eco_names:
        # chose from selected_ids
        eco_sites_id = np.array(
            choose_sites_in_ecoregion(gages, selected_ids, eco_name)
        )
        if eco_sites_id.size < split_num or eco_sites_id.size < 1:
            continue
        for train, test in kf.split(eco_sites_id):
            sites_lst_train.append(eco_sites_id[train])
            sites_lst_test.append(eco_sites_id[test])
            eco_name_chosen.append(eco_name)
    if save:
        if kfold_dir is None:
            raise NotADirectoryError(
                "Please set saved directory and give it to kfold_dir"
            )
        if not os.path.isdir(kfold_dir):
            os.makedirs(kfold_dir)
    final_sites_ids_train_lst = []
    final_sites_ids_test_lst = []
    for i in range(split_num):
        sites_ids_train_ilst = [
            sites_lst_train[j]
            for j in range(len(sites_lst_train))
            if j % split_num == i
        ]
        sites_ids_train_i = np.sort(
            reduce(lambda x, y: np.hstack((x, y)), sites_ids_train_ilst)
        )
        sites_ids_test_ilst = [
            sites_lst_test[j] for j in range(len(sites_lst_test)) if j % split_num == i
        ]
        sites_ids_test_i = np.sort(
            reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst)
        )
        if save:
            kfold_i_train_df = pd.DataFrame({"GAGE_ID": sites_ids_train_i})
            kfold_i_test_df = pd.DataFrame({"GAGE_ID": sites_ids_test_i})
            kfold_i_train_df.to_csv(
                os.path.join(kfold_dir, f"camels_train_kfold{str(i)}.csv"),
                quoting=csv.QUOTE_NONNUMERIC,
                index=None,
            )
            kfold_i_test_df.to_csv(
                os.path.join(kfold_dir, f"camels_test_kfold{str(i)}.csv"),
                quoting=csv.QUOTE_NONNUMERIC,
                index=None,
            )
        final_sites_ids_train_lst.append(sites_ids_train_i.tolist())
        final_sites_ids_test_lst.append(sites_ids_test_i.tolist())
    return final_sites_ids_train_lst, final_sites_ids_test_lst


def split_two_samples_for_ecoregions(
    selected_ids1: list, selected_ids2: list, split_num, save=False, kfold_dir=None
):
    """split each sample of two to training and test

    Parameters
    ----------
    selected_ids1 : list
        the first basins' ids
    selected_ids2 : list
        the second basins' ids
    split_num:
        the k of k-fold cross validation
    save : bool
        if save, training and test basins ids will be saved as csv files
    kfold_dir: str
        the saved directory, if save

    Returns
    -------
    tuple
        (list(training basins ids), list(testing basins ids))
    """
    random_seed = 1234
    eco_names = [
        ("ECO2_CODE", 5.2),
        ("ECO2_CODE", 5.3),
        ("ECO2_CODE", 6.2),
        ("ECO2_CODE", 7.1),
        ("ECO2_CODE", 8.1),
        ("ECO2_CODE", 8.2),
        ("ECO2_CODE", 8.3),
        ("ECO2_CODE", 8.4),
        ("ECO2_CODE", 8.5),
        ("ECO2_CODE", 9.2),
        ("ECO2_CODE", 9.3),
        ("ECO2_CODE", 9.4),
        ("ECO2_CODE", 9.5),
        ("ECO2_CODE", 9.6),
        ("ECO2_CODE", 10.1),
        ("ECO2_CODE", 10.2),
        ("ECO2_CODE", 10.4),
        ("ECO2_CODE", 11.1),
        ("ECO2_CODE", 12.1),
        ("ECO2_CODE", 13.1),
    ]
    np.random.seed(random_seed)
    kf = KFold(n_splits=split_num, shuffle=True, random_state=random_seed)
    # eco attr exists in Gages
    gages_data_path = os.path.join(definitions.DATASET_DIR, "gages")
    gages = Gages(gages_data_path)
    all_index_lst_train_1 = []
    # all sites come from train1 dataset
    sites_lst_train = []
    all_index_lst_test_1 = []
    sites_lst_test_1 = []
    all_index_lst_test_2 = []
    sites_lst_test_2 = []
    eco_name_chosen = []
    for eco_name in eco_names:
        # chose from selected_ids
        train_sites_id_inter = np.array(
            choose_sites_in_ecoregion(gages, selected_ids1, eco_name)
        )
        test_sites_id_inter = np.array(
            choose_sites_in_ecoregion(gages, selected_ids2, eco_name)
        )
        if train_sites_id_inter.size < split_num or test_sites_id_inter.size < 1:
            continue
        for train, test in kf.split(train_sites_id_inter):
            all_index_lst_train_1.append(train)
            sites_lst_train.append(train_sites_id_inter[train])
            all_index_lst_test_1.append(test)
            sites_lst_test_1.append(train_sites_id_inter[test])
            if test_sites_id_inter.size < test.size:
                all_index_lst_test_2.append(np.arange(test_sites_id_inter.size))
                sites_lst_test_2.append(test_sites_id_inter)
            else:
                test2_chosen_idx = np.random.choice(
                    test_sites_id_inter.size, test.size, replace=False
                )
                all_index_lst_test_2.append(test2_chosen_idx)
                sites_lst_test_2.append(test_sites_id_inter[test2_chosen_idx])
        eco_name_chosen.append(eco_name)
    if save:
        if kfold_dir is None:
            raise NotADirectoryError(
                "Please set saved directory and give it to kfold_dir"
            )
        if not os.path.isdir(kfold_dir):
            os.makedirs(kfold_dir)
    final_sites_ids_train_lst = []
    final_sites_ids_test_lst = []
    final_sites_ids_pub_test_lst = []
    for i in range(split_num):
        sites_ids_train_ilst = [
            sites_lst_train[j]
            for j in range(len(sites_lst_train))
            if j % split_num == i
        ]
        sites_ids_train_i = np.sort(
            reduce(lambda x, y: np.hstack((x, y)), sites_ids_train_ilst)
        )
        sites_ids_test_ilst_1 = [
            sites_lst_test_1[j]
            for j in range(len(sites_lst_test_1))
            if j % split_num == i
        ]
        sites_ids_test_i_1 = np.sort(
            reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst_1)
        )
        sites_ids_test_ilst_2 = [
            sites_lst_test_2[j]
            for j in range(len(sites_lst_test_2))
            if j % split_num == i
        ]
        sites_ids_test_i_2 = np.sort(
            reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst_2)
        )
        final_sites_ids_train_lst.append(sites_ids_train_i.tolist())
        final_sites_ids_test_lst.append(sites_ids_test_i_1.tolist())
        final_sites_ids_pub_test_lst.append(sites_ids_test_i_2.tolist())
    return (
        final_sites_ids_train_lst,
        final_sites_ids_test_lst,
        final_sites_ids_pub_test_lst,
    )


def mix_two_samples_for_ecoregions(
    selected_ids1: list, selected_ids2: list, split_num, save=False, kfold_dir=None
):
    """split and mix each sample of two to training and test

    Parameters
    ----------
    selected_ids1 : list
        the first basins' ids
    selected_ids2 : list
        the second basins' ids
    split_num:
        the k of k-fold cross validation
    save : bool
        if save, training and test basins ids will be saved as csv files
    kfold_dir: str
        the saved directory, if save

    Returns
    -------
    tuple
        (list(training basins ids), list(testing basins ids))
    """
    random_seed = 1234
    eco_names = [
        ("ECO2_CODE", 5.2),
        ("ECO2_CODE", 5.3),
        ("ECO2_CODE", 6.2),
        ("ECO2_CODE", 7.1),
        ("ECO2_CODE", 8.1),
        ("ECO2_CODE", 8.2),
        ("ECO2_CODE", 8.3),
        ("ECO2_CODE", 8.4),
        ("ECO2_CODE", 8.5),
        ("ECO2_CODE", 9.2),
        ("ECO2_CODE", 9.3),
        ("ECO2_CODE", 9.4),
        ("ECO2_CODE", 9.5),
        ("ECO2_CODE", 9.6),
        ("ECO2_CODE", 10.1),
        ("ECO2_CODE", 10.2),
        ("ECO2_CODE", 10.4),
        ("ECO2_CODE", 11.1),
        ("ECO2_CODE", 12.1),
        ("ECO2_CODE", 13.1),
    ]
    np.random.seed(random_seed)
    kf = KFold(n_splits=split_num, shuffle=True, random_state=random_seed)
    # eco attr exists in Gages
    gages_data_path = os.path.join(definitions.DATASET_DIR, "gages")
    gages = Gages(gages_data_path)

    sites_lst_train = []
    sites_lst_test_1 = []
    sites_lst_test_2 = []

    eco_name_chosen = []
    for eco_name in eco_names:
        sites_id_inter_1 = np.array(
            choose_sites_in_ecoregion(gages, selected_ids1, eco_name)
        )
        sites_id_inter_2 = np.array(
            choose_sites_in_ecoregion(gages, selected_ids2, eco_name)
        )

        if sites_id_inter_1.size < sites_id_inter_2.size:
            if sites_id_inter_1.size < split_num:
                continue
            for train, test in kf.split(sites_id_inter_1):
                sites_lst_train_1 = sites_id_inter_1[train]
                sites_lst_test_1.append(sites_id_inter_1[test])

                chosen_lst_2 = random_choice_no_return(
                    sites_id_inter_2, [train.size, test.size]
                )
                sites_lst_train_2 = chosen_lst_2[0]
                sites_lst_test_2.append(chosen_lst_2[1])

                sites_lst_train.append(
                    np.sort(np.append(sites_lst_train_1, sites_lst_train_2))
                )

        else:
            if sites_id_inter_2.size < split_num:
                continue
            for train, test in kf.split(sites_id_inter_2):
                sites_lst_train_2 = sites_id_inter_2[train]
                sites_lst_test_2.append(sites_id_inter_2[test])

                chosen_lst_1 = random_choice_no_return(
                    sites_id_inter_1, [train.size, test.size]
                )
                sites_lst_train_1 = chosen_lst_1[0]
                sites_lst_test_1.append(chosen_lst_1[1])

                sites_lst_train.append(
                    np.sort(np.append(sites_lst_train_1, sites_lst_train_2))
                )

        eco_name_chosen.append(eco_name)
    final_sites_ids_train_lst = []
    final_sites_ids_pub1_lst = []
    final_sites_ids_pub2_lst = []
    for i in range(split_num):
        sites_ids_train_ilst = [
            sites_lst_train[j]
            for j in range(len(sites_lst_train))
            if j % split_num == i
        ]
        sites_ids_train_i = np.sort(
            reduce(lambda x, y: np.hstack((x, y)), sites_ids_train_ilst)
        )
        sites_ids_test_ilst_1 = [
            sites_lst_test_1[j]
            for j in range(len(sites_lst_test_1))
            if j % split_num == i
        ]
        sites_ids_test_i_1 = np.sort(
            reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst_1)
        )

        sites_ids_test_ilst_2 = [
            sites_lst_test_2[j]
            for j in range(len(sites_lst_test_2))
            if j % split_num == i
        ]
        sites_ids_test_i_2 = np.sort(
            reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst_2)
        )
        final_sites_ids_train_lst.append(sites_ids_train_i.tolist())
        final_sites_ids_pub1_lst.append(sites_ids_test_i_1.tolist())
        final_sites_ids_pub2_lst.append(sites_ids_test_i_2.tolist())
    return final_sites_ids_train_lst, final_sites_ids_pub1_lst, final_sites_ids_pub2_lst


def read_tb_log(
    a_exp, best_batchsize, exp_example="gages", where_save="transfer_learning"
):
    """Copy a recent log file to the current directory and read the log file.

    Parameters
    ----------
    a_exp : _type_
        _description_
    best_batchsize : _type_
        _description_
    exp_example : str, optional
        _description_, by default "gages"
    where_save : str, optional
        A directory in "app" directory, by default "transfer_learning"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """
    log_dir = os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        exp_example,
        a_exp,
        f"opt_Adadelta_lr_1.0_bsize_{str(best_batchsize)}",
    )
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log dir {log_dir} not found!")
    result_dir = os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "app",
        where_save,
        "results",
        "tensorboard",
        a_exp,
        f"opt_Adadelta_lr_1.0_bsize_{str(best_batchsize)}",
    )
    copy_latest_tblog_file(log_dir, result_dir)
    scalar_file = os.path.join(result_dir, "scalars.csv")
    if not os.path.exists(scalar_file):
        reader = SummaryReader(result_dir)
        df_scalar = reader.scalars
        df_scalar.to_csv(scalar_file, index=False)
    else:
        df_scalar = pd.read_csv(scalar_file)

    # reader = SummaryReader(result_dir)
    histgram_file = os.path.join(result_dir, "histograms.pkl")
    if not os.path.exists(histgram_file):
        reader = SummaryReader(result_dir, pivot=True)
        df_histgram = reader.histograms
        # https://www.statology.org/pandas-save-dataframe/
        df_histgram.to_pickle(histgram_file)
    else:
        df_histgram = pd.read_pickle(histgram_file)
    return df_scalar, df_histgram


def get_latest_event_file(event_file_lst):
    """Get the latest event file in the current directory.

    Returns
    -------
    str
        The latest event file.
    """
    event_files = [Path(f) for f in event_file_lst]
    event_file_names_lst = [event_file.stem.split(".") for event_file in event_files]
    ctimes = [
        int(event_file_names[event_file_names.index("tfevents") + 1])
        for event_file_names in event_file_names_lst
    ]
    return event_files[ctimes.index(max(ctimes))]


# Prepare temp dirs for storing event files
def copy_latest_tblog_file(log_dir, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        copy_lst = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.startswith("events")
        ]
        copy_file = get_latest_event_file(copy_lst)
        shutil.copy(copy_file, result_dir)
