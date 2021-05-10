import argparse
import json
import os
import pandas as pd

import definitions


def default_config_file(data_source_path, dataset_name):
    """The default config file for all models/datasets/training parameters in this repo"""
    config_default = {
        "model_params": {
            # now only PyTorch is supported
            "model_type": "PyTorch",
            # supported models can be seen in hydroDL/model_dict_function.py
            "model_name": "LSTM",
            # the details of model parameters for the "model_name" model
            "model_param": {
                # the rho in LSTM
                "seq_length": 30,
                # the size of input (feature number)
                "n_time_series": 24,
                # the length of output time-sequence
                "output_seq_len": 1,
                "hidden_states": 20,
                "num_layers": 1,
                "bias": True,
                "batch_size": 100,
                "probabilistic": False
            }
        },
        "dataset_params": {
            "dataset_name": dataset_name,
            "download": True,
            "cache_read": True,
            "cache_write": False,
            "cache_path": None,
            "data_path": data_source_path,
            "validation_path": None,
            "test_path": None,
            "batch_size": 100,
            # the rho in LSTM
            "forecast_history": 30,
            "forecast_length": 1,
            # modeled objects
            "object_ids": "ALL",
            # modeling time range
            "t_range_train": ["1992-01-01", "1993-01-01"],
            "t_range_valid": None,
            "t_range_test": ["1993-01-01", "1994-01-01"],
            # the output
            "target_cols": ["usgsFlow"],
            # the time series input
            "relevant_types": ["daymet"],
            "relevant_cols": ["dayl", "prcp", 'srad', 'swe', 'tmax', 'tmin', 'vp'],
            # the attribute input
            "constant_cols": ['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max', 'lai_diff',
                              'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50', 'soil_depth_statsgo',
                              'soil_porosity', 'soil_conductivity', 'max_water_content', 'geol_1st_class',
                              'geol_2nd_class', 'geol_porostiy', 'geol_permeability'],
            # more other cols, use dict to express!
            "other_cols": None,
            # data_loader for loading data to models
            "data_loader": "StreamflowDataset",
            # only numerical scaler: for categorical vars, they are transformed to numerical vars when reading them
            "scaler": "StandardScaler"
        },
        "training_params": {
            # if train_mode is False, don't train and evaluate
            "train_mode": True,
            "criterion": "RMSE",
            "optimizer": "Adam",
            "optim_params": {
                "lr": 0.001,
            },
            "epochs": 10,
            # save_epoch ==0 means only save once in the final epoch
            "save_epoch": 0,
            "batch_size": 100,
            "random_seed": 1234
        },
        # For evaluation
        "metrics": ["NSE"],
    }
    return config_default


def cmd(sub=None, download=0, scaler=None, data_loader=None, rs=None, gage_id_file=None, gage_id=None,
        train_period=None, test_period=None, opt=None, cache_read=None, cache_write=None, cache_path=None,
        hidden_size=None, opt_param=None, batch_size=None, rho=None, train_mode=None, train_epoch=None, save_epoch=None,
        te=None, model_name=None, weight_path=None, continue_train=None, var_c=None, var_t=None, n_feature=None,
        loss_func=None, model_param=None, weight_path_add=None, var_t_type=None, var_o=None, gage_id_screen=None):
    """input args from cmd
    """
    parser = argparse.ArgumentParser(description='Train a Time-Series Deep Learning Model for Basins')
    parser.add_argument('--sub', dest='sub', help='subset and sub experiment', default=sub, type=str)
    parser.add_argument('--download', dest='download', help='Do we need to download data', default=download, type=int)
    parser.add_argument('--scaler', dest='scaler', help='Choose a Scaler function', default=scaler, type=str)
    parser.add_argument('--data_loader', dest='data_loader', help='Choose a data loader class', default=data_loader,
                        type=str)
    parser.add_argument('--ctx', dest='ctx',
                        help='Running Context -- gpu num. E.g `--ctx 0` means run code in the context of gpu 0',
                        type=int, default=None)
    parser.add_argument('--rs', dest='rs', help='random seed', default=rs, type=int)
    parser.add_argument('--te', dest='te', help='test epoch', default=te, type=int)
    # There is something wrong with "bool", so I used 1 as True, 0 as False
    parser.add_argument('--train_mode', dest='train_mode', help='train or test', default=train_mode, type=int)
    parser.add_argument('--train_epoch', dest='train_epoch', help='epoches of training period', default=train_epoch,
                        type=int)
    parser.add_argument('--save_epoch', dest='save_epoch', help='save for every save_epoch epoches', default=save_epoch,
                        type=int)
    parser.add_argument('--loss_func', dest='loss_func', help='choose loss function', default=loss_func, type=str)
    parser.add_argument('--train_period', dest='train_period', help='The training period', default=train_period,
                        nargs='+')
    parser.add_argument('--test_period', dest='test_period', help='The test period', default=test_period, nargs='+')
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=batch_size, type=int)
    parser.add_argument('--rho', dest='rho', help='length of time sequence when training', default=rho, type=int)
    parser.add_argument('--model_name', dest='model_name', help='The name of DL model. now in the zoo',
                        default=model_name, type=str)
    parser.add_argument('--weight_path', dest='weight_path', help='The weights of trained model', default=weight_path,
                        type=str)
    parser.add_argument('--weight_path_add', dest='weight_path_add',
                        help='More info about the weights of trained model', default=weight_path_add, type=json.loads)
    parser.add_argument('--continue_train', dest='continue_train',
                        help='Continue to train the model from weight_path when continue_train>0',
                        default=continue_train, type=int)
    parser.add_argument('--gage_id', dest='gage_id', help='just select some sites', default=gage_id, nargs='+')
    parser.add_argument('--gage_id_screen', dest='gage_id_screen', help='the criterion to chose some gages',
                        default=gage_id_screen, type=json.loads)
    parser.add_argument('--gage_id_file', dest='gage_id_file', help='select some sites from a file',
                        default=gage_id_file, type=str)
    parser.add_argument('--opt', dest='opt', help='choose an optimizer', default=opt, type=str)
    parser.add_argument('--opt_param', dest='opt_param', help='the optimizer parameters', default=opt_param,
                        type=json.loads)
    parser.add_argument('--var_c', dest='var_c', help='types of attributes', default=var_c, nargs='+')
    parser.add_argument('--var_t', dest='var_t', help='types of forcing', default=var_t, nargs='+')
    parser.add_argument('--var_t_type', dest='var_t_type', help='types of forcing dataset', default=var_t_type,
                        nargs='+')
    parser.add_argument('--var_o', dest='var_o', help='more other inputs except for var_c and var_t', default=var_o,
                        type=json.loads)
    parser.add_argument('--n_feature', dest='n_feature', help='the number of features', default=n_feature, type=int)
    parser.add_argument('--cache_read', dest='cache_read', help='read binary file', default=cache_read, type=int)
    parser.add_argument('--cache_write', dest='cache_write', help='write binary file', default=cache_write, type=int)
    parser.add_argument('--cache_path', dest='cache_path', help='specify the directory of data cache files',
                        default=cache_path, type=str)
    parser.add_argument('--hidden_size', dest='hidden_size', help='the hidden_size of nn', default=hidden_size,
                        type=int)
    parser.add_argument('--model_param', dest='model_param', help='the model_param in model_params',
                        default=model_param, type=json.loads)
    args = parser.parse_args()
    return args


def update_cfg(cfg_file, new_args):
    print("update config file")
    if new_args.sub is not None:
        subset, subexp = new_args.sub.split("/")
        if not os.path.exists(os.path.join(definitions.ROOT_DIR, "example", subset, subexp)):
            os.makedirs(os.path.join(definitions.ROOT_DIR, "example", subset, subexp))
        cfg_file["dataset_params"]["validation_path"] = os.path.join(definitions.ROOT_DIR, "example", subset, subexp)
        cfg_file["dataset_params"]["test_path"] = os.path.join(definitions.ROOT_DIR, "example", subset, subexp)
        if new_args.cache_path is not None:
            cfg_file["dataset_params"]["cache_path"] = new_args.cache_path
        else:
            cfg_file["dataset_params"]["cache_path"] = os.path.join(definitions.ROOT_DIR, "example", subset, subexp)
    if new_args.download is not None:
        if new_args.download == 0:
            cfg_file["dataset_params"]["download"] = False
        else:
            cfg_file["dataset_params"]["download"] = True
    if new_args.scaler is not None:
        cfg_file["dataset_params"]["scaler"] = new_args.scaler
    if new_args.data_loader is not None:
        cfg_file["dataset_params"]["data_loader"] = new_args.data_loader
    if new_args.ctx is not None:
        cfg_file.CTX = new_args.ctx
    if new_args.rs is not None:
        cfg_file["training_params"]["random_seed"] = new_args.rs
    if new_args.te is not None:
        cfg_file.TEST_EPOCH = new_args.te
    if new_args.train_mode is not None:
        if new_args.train_mode > 0:
            cfg_file["training_params"]["train_mode"] = True
        else:
            cfg_file["training_params"]["train_mode"] = False
    if new_args.loss_func is not None:
        cfg_file["training_params"]["criterion"] = new_args.loss_func
    if new_args.train_period is not None:
        cfg_file["dataset_params"]["t_range_train"] = new_args.train_period
    if new_args.test_period is not None:
        cfg_file["dataset_params"]["t_range_test"] = new_args.test_period
    if new_args.gage_id is not None or new_args.gage_id_file is not None:
        if new_args.gage_id_file is not None:
            gage_id_lst = pd.read_csv(new_args.gage_id_file, dtype={0: str}).iloc[:, 0].values
            cfg_file["dataset_params"]["object_ids"] = gage_id_lst.tolist()
        else:
            cfg_file["dataset_params"]["object_ids"] = new_args.gage_id
    if new_args.opt is not None:
        cfg_file["training_params"]["optimizer"] = new_args.opt
        if new_args.opt_param is not None:
            cfg_file["training_params"]["optim_params"] = new_args.opt_param
        else:
            cfg_file["training_params"]["optim_params"] = {}
    if new_args.var_c is not None:
        cfg_file["dataset_params"]["constant_cols"] = new_args.var_c
    if new_args.var_t is not None:
        cfg_file["dataset_params"]["relevant_cols"] = new_args.var_t
    if new_args.var_t_type is not None:
        cfg_file["dataset_params"]["relevant_types"] = new_args.var_t_type
    if new_args.var_o is not None:
        cfg_file["dataset_params"]["other_cols"] = new_args.var_o
    if new_args.train_epoch is not None:
        cfg_file["training_params"]["epochs"] = new_args.train_epoch
    if new_args.save_epoch is not None:
        cfg_file["training_params"]["save_epoch"] = new_args.save_epoch
    if new_args.cache_read is not None:
        if new_args.cache_read > 0:
            cfg_file["dataset_params"]["cache_read"] = True
        else:
            cfg_file["dataset_params"]["cache_read"] = False
    if new_args.cache_write is not None:
        if new_args.cache_write > 0:
            cfg_file["dataset_params"]["cache_write"] = True
            assert cfg_file["dataset_params"]["cache_read"]
        else:
            cfg_file["dataset_params"]["cache_write"] = False
    if new_args.model_name is not None:
        cfg_file["model_params"]["model_name"] = new_args.model_name
    if new_args.weight_path is not None:
        cfg_file["model_params"]["weight_path"] = new_args.weight_path
        if new_args.continue_train is None or new_args.continue_train == 0:
            continue_train = False
        else:
            continue_train = True
        cfg_file["model_params"]["continue_train"] = continue_train
    if new_args.weight_path_add is not None:
        cfg_file["model_params"]["weight_path_add"] = new_args.weight_path_add
    if new_args.model_param is None:
        if new_args.batch_size is not None:
            batch_size = new_args.batch_size
            cfg_file["model_params"]["model_param"]["batch_size"] = batch_size
            cfg_file["dataset_params"]["batch_size"] = batch_size
            cfg_file["training_params"]["batch_size"] = batch_size
        if new_args.rho is not None:
            rho = new_args.rho
            cfg_file["model_params"]["model_param"]["seq_length"] = rho
            cfg_file["dataset_params"]["forecast_history"] = rho
        if new_args.n_feature is not None:
            cfg_file["model_params"]["model_param"]["n_time_series"] = new_args.n_feature
        if new_args.hidden_size is not None:
            cfg_file["model_params"]["model_param"]["hidden_states"] = new_args.hidden_size
    else:
        cfg_file["model_params"]["model_param"] = new_args.model_param
        if "batch_size" in new_args.model_param.keys():
            cfg_file["dataset_params"]["batch_size"] = new_args.model_param["batch_size"]
            cfg_file["training_params"]["batch_size"] = new_args.model_param["batch_size"]
        elif new_args.batch_size is not None:
            batch_size = new_args.batch_size
            cfg_file["dataset_params"]["batch_size"] = batch_size
            cfg_file["training_params"]["batch_size"] = batch_size
        if "seq_length" in new_args.model_param.keys():
            cfg_file["dataset_params"]["forecast_history"] = new_args.model_param["seq_length"]
        elif "forecast_history" in new_args.model_param.keys():
            cfg_file["dataset_params"]["forecast_history"] = new_args.model_param["forecast_history"]
        elif new_args.rho is not None:
            cfg_file["dataset_params"]["forecast_history"] = new_args.rho
    print("the updated config:\n", json.dumps(cfg_file, indent=4, ensure_ascii=False))
