import os
import torch
import sys

sys.path.append("../..")
from data.data_camels import Camels
import definitions
from data.config import default_config_file, cmd, update_cfg
from hydroDL.trainer import train_and_evaluate
from data.cache.cache_factory import cache_dataset


def main(config_data, args):
    """
    Main function which is called from the command line. Entrypoint for training all ML models.
    """
    update_cfg(config_data, args)
    if config_data["dataset_params"]["cache_write"]:
        dataset_params = config_data["dataset_params"]
        dataset = Camels(dataset_params["data_path"], dataset_params["download"])
        cache_dataset(dataset_params, dataset)
    with torch.cuda.device(0):
        train_and_evaluate(config_data)
    print("All processes are finished!")


# python camels671_analysis.py --sub test/exp1 --download 0 --model_name KuaiLSTM --opt Adadelta --rs 1234 --cache_write 1 --scaler DapengScaler --data_loader StreamflowDataModel --batch_size 5 --rho 20 --n_feature 24 --gage_id 01013500 01022500 01030500 01031500 01047000 01052500 01054200 01055000 01057000 01170100
# python camels671_analysis.py --sub camels/cache-671sites-19851001_19951001_19951001_20051001-17attr-6forcing --download 0 --model_name KuaiLSTM --opt Adadelta --loss_func RMSESum --hidden_size 256 --rs 1234 --cache_write 1 --train_mode 0 --train_period 1985-10-01 1995-10-01 --test_period 1995-10-01 2005-10-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 50 --batch_size 100 --rho 365 --var_t dayl prcp srad tmax tmin vp --n_feature 23
# python camels671_analysis.py --sub camels/exp2 --download 0 --model_name KuaiLSTM --opt Adadelta --loss_func RMSESum  --hidden_size 256 --rs 1234 --cache_read 1 --cache_path /mnt/data/owen411/code/hydro-spdb-dl/example/camels/cache-671sites-19851001_19951001_19951001_20051001-17attr-6forcing --train_period 1985-10-01 1995-10-01 --test_period 1995-10-01 2005-10-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 50 --batch_size 100 --rho 365 --var_t dayl prcp srad tmax tmin vp --n_feature 23
if __name__ == '__main__':
    print("Begin\n")
    camels_dir = os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "camels")
    dataset_name = "CAMELS"
    config = default_config_file(camels_dir, dataset_name)
    cmd_args = cmd()
    main(config, cmd_args)
    print("End\n")
