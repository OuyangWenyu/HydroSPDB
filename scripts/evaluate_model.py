"""
Author: Wenyu Ouyang
Date: 2023-04-05 20:57:26
LastEditTime: 2023-04-20 22:50:01
LastEditors: Wenyu Ouyang
Description: Evaluate the trained model
FilePath: /HydroSPDB/scripts/evaluate_model.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import json
import glob
import argparse
import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from scripts.streamflow_utils import predict_new_gages_exp
from hydrospdb.data.source.data_constant import Q_CAMELS_US_NAME


def evaluate(args):
    weight_path = args.weight_path
    if weight_path is None:
        raise ValueError("weight_path is required")
    train_exp_dir = os.sep.join(weight_path.split("/")[:-1])
    stat_dict_file = glob.glob(os.path.join(train_exp_dir, "*_stat.json"))[0]
    exp = args.exp
    train_info_file = os.path.join(train_exp_dir, "train_data_dict.json")
    with open(train_info_file, 'r') as f:
        # Load the JSON data from the file
        train_info = json.load(f)
    train_periods = train_info["t_final_range"]
    test_periods = args.test_period
    cache_dir = args.cache_path
    if cache_dir is None or cache_dir == "None":
        cache_dir = os.path.join(definitions.RESULT_DIR, "gages", exp)
    predict_new_gages_exp(
        exp=exp,
        continue_train=False,
        weight_path=weight_path,
        train_period=train_periods,
        test_period=test_periods,
        cache_path=cache_dir,
        gage_id_file=os.path.join(definitions.RESULT_DIR, "gage_id.csv"),
        stat_dict_file=stat_dict_file,
    )


if __name__ == "__main__":
    print("Here are your commands:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        dest="exp",
        help="the ID of the experiment, such as exp0010",
        type=str,
        default="exp0010",
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="testing period, such as ['2011-10-01', '2016-10-01']",
        nargs="+",
        default=["2016-10-01", "2019-10-01"],
    )
    parser.add_argument(
        "--cache_path",
        dest="cache_path",
        help="the cache file for forcings, attributes and targets data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--weight_path",
        dest="weight_path",
        help="the weight path file for trained model",
        type=str,
        # default=None,
        default="/mnt/sdc/owen/code/HydroSPDB/results/gages/exp001/20_April_202306_45PM_model.pth",
    )
    args = parser.parse_args()
    print(f"Your command arguments:{str(args)}")
    evaluate(args)
