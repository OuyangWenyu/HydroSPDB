"""1. zero-dor and small-dor basins 2. large-dor basins"""
import os
import pandas as pd
import torch
import sys

sys.path.append("../..")
import definitions
from data.cache.cache_factory import cache_dataset
from data.config import update_cfg, default_config_file, cmd
from data.pro.data_gages_pro import GagesPro
from hydroDL.trainer import train_and_evaluate
from data.pro.select_gages_ids import dor_reservoirs_chosen
from data.data_gages import Gages


def main(config_data, args, gage_id_screen):
    """
    Main function which is called from the command line. Entrypoint for training all ML models.
    """
    if args.download is not None:
        if args.download == 0:
            download = False
        else:
            download = True
    else:
        download = False
    gages = Gages(config_data["dataset_params"]["data_path"][1], download)
    if args.gage_id is not None or args.gage_id_file is not None:
        if args.gage_id_file is not None:
            gage_id_lst = pd.read_csv(args.gage_id_file, dtype={0: str}).iloc[:, 0].values
            usgs_ids = gage_id_lst.tolist()
        else:
            usgs_ids = args.gage_id
    else:
        usgs_ids = gages.read_object_ids()
    if gage_id_screen is not None:
        if "DOR" in gage_id_screen.keys():
            chosen_ids = dor_reservoirs_chosen(gages, usgs_ids, gage_id_screen["DOR"])
        else:
            raise NotImplementedError("NO such choice yet!")
        args.gage_id = chosen_ids
        args.gage_id_file = None

    update_cfg(config_data, args)

    if config_data["dataset_params"]["cache_write"]:
        dataset_params = config_data["dataset_params"]
        dataset = GagesPro(dataset_params["data_path"], dataset_params["download"])
        cache_dataset(dataset_params, dataset)

    with torch.cuda.device(0):
        train_and_evaluate(config_data)
    print("Process is now complete.")


# python gages_dor_analysis.py --sub dam/cache-dorminus0.1-19900101_20000101_20000101_20100101-30attr-7forcing --gage_id_file /mnt/data/owen411/code/hydro-spdb-dl/example/3557basins_ID_NSE_DOR.csv --gage_id_screen {\"DOR\":-0.1} --download 0 --model_name KuaiLSTM --opt Adadelta --loss_func RMSESum --hidden_size 256 --rs 1234 --train_mode 0 --cache_write 1 --train_period 1990-01-01 2000-01-01 --test_period 2000-01-01 2010-01-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 50 --batch_size 100 --rho 365 --var_c DRAIN_SQKM ELEV_MEAN_M_BASIN SLOPE_PCT DEVNLCD06 FORESTNLCD06 PLANTNLCD06 WATERNLCD06 SNOWICENLCD06 BARRENNLCD06 SHRUBNLCD06 GRASSNLCD06 WOODYWETNLCD06 EMERGWETNLCD06 AWCAVE PERMAVE RFACT ROCKDEPAVE GEOL_REEDBUSH_DOM GEOL_REEDBUSH_DOM_PCT STREAMS_KM_SQ_KM NDAMS_2009 STOR_NOR_2009 RAW_DIS_NEAREST_MAJ_DAM CANALS_PCT RAW_DIS_NEAREST_CANAL FRESHW_WITHDRAWAL POWER_SUM_MW PDEN_2000_BLOCK ROADS_KM_SQ_KM IMPNLCD06 --n_feature 37
# python gages_dor_analysis.py --sub dam/exp2 --gage_id_file /mnt/data/owen411/code/hydro-spdb-dl/example/3557basins_ID_NSE_DOR.csv --gage_id_screen {\"DOR\":-0.1} --download 0 --model_name KuaiLSTM --opt Adadelta --loss_func RMSESum --hidden_size 256 --rs 1234 --cache_read 1 --cache_path /mnt/data/owen411/code/hydro-spdb-dl/example/dam/cache-dorminus0.1-19900101_20000101_20000101_20100101-30attr-7forcing --train_period 1990-01-01 2000-01-01 --test_period 2000-01-01 2010-01-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 50 --batch_size 100 --rho 365 --var_c DRAIN_SQKM ELEV_MEAN_M_BASIN SLOPE_PCT DEVNLCD06 FORESTNLCD06 PLANTNLCD06 WATERNLCD06 SNOWICENLCD06 BARRENNLCD06 SHRUBNLCD06 GRASSNLCD06 WOODYWETNLCD06 EMERGWETNLCD06 AWCAVE PERMAVE RFACT ROCKDEPAVE GEOL_REEDBUSH_DOM GEOL_REEDBUSH_DOM_PCT STREAMS_KM_SQ_KM NDAMS_2009 STOR_NOR_2009 RAW_DIS_NEAREST_MAJ_DAM CANALS_PCT RAW_DIS_NEAREST_CANAL FRESHW_WITHDRAWAL POWER_SUM_MW PDEN_2000_BLOCK ROADS_KM_SQ_KM IMPNLCD06 --n_feature 37
# python gages_dor_analysis.py --sub dam/cache-dorplus0.1-19900101_20000101_20000101_20100101-30attr-7forcing --gage_id_file /mnt/data/owen411/code/hydro-spdb-dl/example/3557basins_ID_NSE_DOR.csv --gage_id_screen {\"DOR\":0.1} --download 0 --model_name LSTMKernel --opt Adadelta --loss_func RMSESum --hidden_size 256 --rs 1234 --weight_path_add {\"freeze_params\":[\"linearIn.bias\"\,\"linearIn.weight\"\,\"linearOut.bias\"\,\"linearOut.weight\"\,\"lstm.b_hh\"\,\"lstm.b_ih\"\,\"lstm.w_hh\"\,\"lstm.w_ih\"]} --weight_path /mnt/data/owen411/code/hydro-spdb-dl/example/dam/exp2/17_April_202101_00AM_model.pth --continue_train 1 --model_param {\"nx\":37\,\"ny\":1\,\"hidden_size\":256\,\"nk\":1\,\"hidden_size_later\":256\,\"cut\":false\,\"dr\":0.5} --train_mode 0 --cache_write 1 --train_period 1990-01-01 2000-01-01 --test_period 2000-01-01 2010-01-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 50 --batch_size 100 --rho 365 --var_c DRAIN_SQKM ELEV_MEAN_M_BASIN SLOPE_PCT DEVNLCD06 FORESTNLCD06 PLANTNLCD06 WATERNLCD06 SNOWICENLCD06 BARRENNLCD06 SHRUBNLCD06 GRASSNLCD06 WOODYWETNLCD06 EMERGWETNLCD06 AWCAVE PERMAVE RFACT ROCKDEPAVE GEOL_REEDBUSH_DOM GEOL_REEDBUSH_DOM_PCT STREAMS_KM_SQ_KM NDAMS_2009 STOR_NOR_2009 RAW_DIS_NEAREST_MAJ_DAM CANALS_PCT RAW_DIS_NEAREST_CANAL FRESHW_WITHDRAWAL POWER_SUM_MW PDEN_2000_BLOCK ROADS_KM_SQ_KM IMPNLCD06 --n_feature 37
# python gages_dor_analysis.py --sub dam/exp3 --gage_id_file /mnt/data/owen411/code/hydro-spdb-dl/example/3557basins_ID_NSE_DOR.csv --gage_id_screen {\"DOR\":0.1} --download 0 --model_name LSTMKernel --opt Adadelta --loss_func RMSESum --hidden_size 256 --rs 1234 --weight_path_add {\"freeze_params\":[\"linearIn.bias\"\,\"linearIn.weight\"\,\"linearOut.bias\"\,\"linearOut.weight\"\,\"lstm.b_hh\"\,\"lstm.b_ih\"\,\"lstm.w_hh\"\,\"lstm.w_ih\"]} --weight_path /mnt/data/owen411/code/hydro-spdb-dl/example/dam/exp2/17_April_202101_00AM_model.pth --continue_train 1 --model_param {\"nx\":37\,\"ny\":1\,\"hidden_size\":256\,\"nk\":1\,\"hidden_size_later\":256\,\"cut\":false\,\"dr\":0.5} --cache_read 1 --cache_path /mnt/data/owen411/code/hydro-spdb-dl/example/dam/cache-dorplus0.1-19900101_20000101_20000101_20100101-30attr-7forcing --train_period 1990-01-01 2000-01-01 --test_period 2000-01-01 2010-01-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 50 --batch_size 100 --rho 365 --var_c DRAIN_SQKM ELEV_MEAN_M_BASIN SLOPE_PCT DEVNLCD06 FORESTNLCD06 PLANTNLCD06 WATERNLCD06 SNOWICENLCD06 BARRENNLCD06 SHRUBNLCD06 GRASSNLCD06 WOODYWETNLCD06 EMERGWETNLCD06 AWCAVE PERMAVE RFACT ROCKDEPAVE GEOL_REEDBUSH_DOM GEOL_REEDBUSH_DOM_PCT STREAMS_KM_SQ_KM NDAMS_2009 STOR_NOR_2009 RAW_DIS_NEAREST_MAJ_DAM CANALS_PCT RAW_DIS_NEAREST_CANAL FRESHW_WITHDRAWAL POWER_SUM_MW PDEN_2000_BLOCK ROADS_KM_SQ_KM IMPNLCD06 --n_feature 37
# python gages_dor_analysis.py --sub dam/exp4 --gage_id_file /mnt/data/owen411/code/hydro-spdb-dl/example/3557basins_ID_NSE_DOR.csv --gage_id_screen {\"DOR\":0.1} --download 0 --model_name LSTMKernel --opt Adadelta --loss_func RMSESum --hidden_size 256 --rs 1234 --weight_path_add {\"freeze_params\":[\"linearIn.bias\"\,\"linearIn.weight\"\,\"linearOut.bias\"\,\"linearOut.weight\"\,\"lstm.b_hh\"\,\"lstm.b_ih\"\,\"lstm.w_hh\"\,\"lstm.w_ih\"]} --weight_path /mnt/data/owen411/code/hydro-spdb-dl/example/dam/exp2/17_April_202101_00AM_model.pth --continue_train 1 --model_param {\"nx\":37\,\"ny\":1\,\"hidden_size\":256\,\"nk\":1\,\"hidden_size_later\":256\,\"cut\":false\,\"dr\":0.5\,\"delta_s\":true} --cache_read 1 --cache_path /mnt/data/owen411/code/hydro-spdb-dl/example/dam/cache-dorplus0.1-19900101_20000101_20000101_20100101-30attr-7forcing --train_period 1990-01-01 2000-01-01 --test_period 2000-01-01 2010-01-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 50 --batch_size 100 --rho 365 --var_c DRAIN_SQKM ELEV_MEAN_M_BASIN SLOPE_PCT DEVNLCD06 FORESTNLCD06 PLANTNLCD06 WATERNLCD06 SNOWICENLCD06 BARRENNLCD06 SHRUBNLCD06 GRASSNLCD06 WOODYWETNLCD06 EMERGWETNLCD06 AWCAVE PERMAVE RFACT ROCKDEPAVE GEOL_REEDBUSH_DOM GEOL_REEDBUSH_DOM_PCT STREAMS_KM_SQ_KM NDAMS_2009 STOR_NOR_2009 RAW_DIS_NEAREST_MAJ_DAM CANALS_PCT RAW_DIS_NEAREST_CANAL FRESHW_WITHDRAWAL POWER_SUM_MW PDEN_2000_BLOCK ROADS_KM_SQ_KM IMPNLCD06 --n_feature 37
# python gages_dor_analysis.py --sub dam/exp5 --gage_id_file /mnt/data/owen411/code/hydro-spdb-dl/example/3557basins_ID_NSE_DOR.csv --gage_id_screen {\"DOR\":0.1} --download 0 --model_name KuaiLSTM --opt Adadelta --loss_func RMSESum --hidden_size 256 --rs 1234 --cache_read 1 --cache_path /mnt/data/owen411/code/hydro-spdb-dl/example/dam/cache-dorplus0.1-19900101_20000101_20000101_20100101-30attr-7forcing --train_period 1990-01-01 2000-01-01 --test_period 2000-01-01 2010-01-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 50 --batch_size 100 --rho 365 --var_c DRAIN_SQKM ELEV_MEAN_M_BASIN SLOPE_PCT DEVNLCD06 FORESTNLCD06 PLANTNLCD06 WATERNLCD06 SNOWICENLCD06 BARRENNLCD06 SHRUBNLCD06 GRASSNLCD06 WOODYWETNLCD06 EMERGWETNLCD06 AWCAVE PERMAVE RFACT ROCKDEPAVE GEOL_REEDBUSH_DOM GEOL_REEDBUSH_DOM_PCT STREAMS_KM_SQ_KM NDAMS_2009 STOR_NOR_2009 RAW_DIS_NEAREST_MAJ_DAM CANALS_PCT RAW_DIS_NEAREST_CANAL FRESHW_WITHDRAWAL POWER_SUM_MW PDEN_2000_BLOCK ROADS_KM_SQ_KM IMPNLCD06 --n_feature 37
if __name__ == '__main__':
    print("Begin\n")
    gages_dir = [os.path.join(definitions.DATASET_DIR, "gages_pro"),
                 os.path.join(definitions.DATASET_DIR, "gages"),
                 os.path.join(definitions.DATASET_DIR, "nid"),
                 os.path.join(definitions.DATASET_DIR, "gridmet")]
    dataset_name = "GAGES_PRO"
    config = default_config_file(gages_dir, dataset_name)
    cmd_args = cmd()
    main(config, cmd_args, cmd_args.gage_id_screen)
    print("End\n")
