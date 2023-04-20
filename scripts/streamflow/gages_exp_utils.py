"""
Author: Wenyu Ouyang
Date: 2023-04-20 11:51:06
LastEditTime: 2023-04-20 18:04:28
LastEditors: Wenyu Ouyang
Description: functions for gages experiments
FilePath: /HydroSPDB/scripts/streamflow/gages_exp_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from scripts.streamflow.script_constant import (
    VAR_C_CHOSEN_FROM_GAGES_II,
    VAR_T_CHOSEN_FROM_DAYMET,
)
from scripts.streamflow.streamflow_utils import get_lastest_weight_path
from hydrospdb.data.config import cmd, default_config_file, update_cfg
from hydrospdb.models.trainer import train_and_evaluate
from hydrospdb.data.source.data_constant import DAYMET_NAME, Q_CAMELS_US_NAME


def run_gages_exp(
    target_exp,
    var_c=VAR_C_CHOSEN_FROM_GAGES_II,
    var_t=VAR_T_CHOSEN_FROM_DAYMET,
    train_period=None,
    test_period=None,
    gage_id_file=os.path.join(
        definitions.RESULT_DIR,
        "gage_id.csv",
    ),
    cache_dir=None,
    random_seed=1234,
    ctx=None,
    loss_func="RMSESum",
):
    if ctx is None:
        ctx = [0]
    if train_period is None:
        train_period = ["2001-10-01", "2011-10-01"]
    if test_period is None:
        test_period = ["2011-10-01", "2016-10-01"]
    weight_path_dir = os.path.join(definitions.RESULT_DIR, target_exp)
    try:
        weight_path = (
            get_lastest_weight_path(weight_path_dir)
            if os.path.exists(weight_path_dir)
            else None
        )
    except Exception as e:
        weight_path = None
    config_data = default_config_file()
    args = cmd(
        sub=f"gages/{target_exp}",
        source="GAGES",
        source_path=definitions.DATASET_DIR,
        download=0,
        ctx=ctx,
        model_name="KuaiLSTM",
        model_param={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        weight_path=weight_path,
        loss_func=loss_func,
        cache_read=1,
        cache_write=1,
        batch_size=100,
        rho=365,
        var_t=var_t,
        var_c=var_c,
        var_t_type=[DAYMET_NAME],
        var_out=[Q_CAMELS_US_NAME],
        train_period=train_period,
        test_period=test_period,
        opt="Adadelta",
        rs=random_seed,
        data_loader="StreamflowDataModel",
        scaler="DapengScaler",
        n_output=1,
        train_epoch=300,
        save_epoch=20,
        te=300,
        # train_epoch=2,
        # save_epoch=1,
        # te=2,
        gage_id_file=gage_id_file,
    )
    update_cfg(config_data, args)
    if cache_dir is not None:
        # train_data_dict.json is a flag for cache existing
        if not os.path.exists(os.path.join(cache_dir, "train_data_dict.json")):
            cache_dir = None
        else:
            config_data["data_params"]["cache_path"] = cache_dir
            config_data["data_params"]["cache_read"] = True
            config_data["data_params"]["cache_write"] = False
    train_and_evaluate(config_data)
    print("All processes are finished!")
