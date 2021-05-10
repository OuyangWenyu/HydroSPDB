import copy
import os
import torch
import pandas as pd
import definitions
from data.cache.cache_factory import cache_dataset
from data.config import default_config_file, cmd, update_cfg
from data.data_gages import Gages
from data.pro.data_gages_pro import GagesPro
from data.pro.select_gages_ids import dor_reservoirs_chosen
from hydroDL.time_model import PyTorchForecast
from hydroDL.custom.dilate_loss import DilateLoss
from hydroDL.pytorch_training import compute_loss
import unittest

from hydroDL.trainer import train_and_evaluate


class PyTorchTrainTests(unittest.TestCase):
    def setUp(self):
        gages_dir = [os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages_pro"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "nid"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gridmet")]
        dataset_name = "GAGES_PRO"
        self.config_data = default_config_file(gages_dir, dataset_name)
        attr_basin = ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'SLOPE_PCT']
        attr_landcover = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'BARRENNLCD06',
                          'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
        attr_soil = ['AWCAVE', 'PERMAVE', 'RFACT', 'ROCKDEPAVE']
        attr_geol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT']
        attr_hydro = ['STREAMS_KM_SQ_KM']
        attr_hydro_mod_dams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_DIS_NEAREST_MAJ_DAM']
        attr_hydro_mod_other = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'FRESHW_WITHDRAWAL', 'POWER_SUM_MW']
        attr_pop_infrastr = ['PDEN_2000_BLOCK', 'ROADS_KM_SQ_KM', 'IMPNLCD06']
        var_c = attr_basin + attr_landcover + attr_soil + attr_geol + attr_hydro + attr_hydro_mod_dams + attr_hydro_mod_other + attr_pop_infrastr

        self.args1 = cmd(sub="gages_gridmet/exp8",
                         download=0,
                         model_name="KaiTlLSTM",
                         opt="Adadelta",
                         rs=1234,
                         cache_read=1,
                         cache_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages_gridmet/cache-328sites-20080101_20130101_20130101_20180101-30attr-8forcing",
                         loss_func="RMSESum",
                         hidden_size=256,
                         scaler="DapengScaler",
                         data_loader="StreamflowDataModel",
                         batch_size=100,
                         rho=365,
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages_gridmet/exp6/13_April_202111_54AM_model.pth",
                         weight_path_add={
                             "freeze_params": ["linearIn.bias", "linearIn.weight", "linearOut.bias", "linearOut.weight",
                                               "lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]},
                         var_c=var_c,
                         n_feature=len(var_c) + 8,
                         train_epoch=20,
                         var_t=["pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto", "cet"],
                         var_t_type="gridmet",
                         train_period=["2008-01-01", "2013-01-01"],
                         test_period=["2013-01-01", "2018-01-01"],
                         gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/328irrigation_gage_id.csv",
                         model_param={"seq_length": 365, "n_time_series": 38, "input_seq_len": 37, "output_seq_len": 1,
                                      "hidden_states": 256, "num_layers": 1, "bias": True, "batch_size": 100,
                                      "probabilistic": False},
                         continue_train=1)

        self.args2 = cmd(sub="gages_gridmet/exp8",
                         download=0,
                         model_name="KaiTlLSTM",
                         opt="Adadelta",
                         rs=1234,
                         cache_read=1,
                         cache_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages_gridmet/cache-328sites-20080101_20130101_20130101_20180101-30attr-8forcing",
                         loss_func="RMSESum",
                         hidden_size=256,
                         scaler="DapengScaler",
                         data_loader="StreamflowDataModel",
                         batch_size=100,
                         rho=365,
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages_gridmet/exp6/13_April_202111_54AM_model.pth",
                         weight_path_add={
                             "freeze_params": ["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]},
                         var_c=var_c,
                         n_feature=len(var_c) + 8,
                         train_epoch=20,
                         var_t=["pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto", "cet"],
                         var_t_type="gridmet",
                         train_period=["2008-01-01", "2013-01-01"],
                         test_period=["2013-01-01", "2018-01-01"],
                         gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/328irrigation_gage_id.csv",
                         model_param={"seq_length": 365, "n_time_series": 38, "input_seq_len": 37, "output_seq_len": 1,
                                      "hidden_states": 256, "num_layers": 1, "bias": True, "batch_size": 100,
                                      "probabilistic": False},
                         continue_train=1)

        self.args3 = cmd(sub="gages_gridmet/exp8",
                         download=0,
                         model_name="KaiTlLSTM",
                         opt="Adadelta",
                         rs=1234,
                         cache_read=1,
                         cache_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages_gridmet/cache-328sites-20080101_20130101_20130101_20180101-30attr-8forcing",
                         loss_func="RMSESum",
                         hidden_size=256,
                         scaler="DapengScaler",
                         data_loader="StreamflowDataModel",
                         batch_size=100,
                         rho=365,
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages_gridmet/exp6/13_April_202111_54AM_model.pth",
                         var_c=var_c,
                         n_feature=len(var_c) + 8,
                         train_epoch=300,
                         var_t=["pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto", "cet"],
                         var_t_type="gridmet",
                         train_period=["2008-01-01", "2013-01-01"],
                         test_period=["2013-01-01", "2018-01-01"],
                         gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/328irrigation_gage_id.csv",
                         model_param={"seq_length": 365, "n_time_series": 38, "input_seq_len": 37, "output_seq_len": 1,
                                      "hidden_states": 256, "num_layers": 1, "bias": True, "batch_size": 100,
                                      "probabilistic": False},
                         continue_train=1)
        self.args4 = cmd(sub="dam/exp1", download=0, model_name="KuaiLSTM", opt="Adadelta", rs=1234, cache_write=1,
                         scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                         var_c=var_c, n_feature=len(var_c) + 7, train_epoch=20,
                         loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                         test_period=["1995-10-01", "1996-10-01"], hidden_size=256,
                         gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                  "01055000", "01057000", "01170100"], gage_id_screen={"DOR": -0.1})
        self.args5 = cmd(sub="dam/exp1", download=0, model_name="LSTMKernel", opt="Adadelta", rs=1234, cache_write=1,
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/dam/exp1/16_April_202108_43PM_model.pth",
                         weight_path_add={
                             "freeze_params": ["linearIn.bias", "linearIn.weight", "linearOut.bias", "linearOut.weight",
                                               "lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]},
                         continue_train=True,
                         model_param={"nx": len(var_c) + 7, "ny": 1, "hidden_size": 256, "nk": 1,
                                      "hidden_size_later": 256, "cut": True, "dr": 0.5},
                         scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                         var_c=var_c, n_feature=len(var_c) + 7, train_epoch=20,
                         loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                         test_period=["1995-10-01", "1996-10-01"], hidden_size=256,
                         gage_id=["01104430", "01175500", "01302000", "01302020", "01399670", "01443900", "02187500",
                                  "02277000", "02292900"],
                         gage_id_screen={"DOR": 0.1})
        self.args6 = cmd(sub="dam/exp1", download=0, model_name="LSTMKernel", opt="Adadelta", rs=1234, cache_write=1,
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/dam/exp1/16_April_202108_43PM_model.pth",
                         weight_path_add={
                             "freeze_params": ["linearIn.bias", "linearIn.weight", "linearOut.bias", "linearOut.weight",
                                               "lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]},
                         continue_train=True,
                         model_param={"nx": len(var_c) + 7, "ny": 1, "hidden_size": 256, "nk": 1,
                                      "hidden_size_later": 256, "cut": False, "dr": 0.5, "delta_s": True},
                         scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                         var_c=var_c, n_feature=len(var_c) + 7, train_epoch=20,
                         loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                         test_period=["1995-10-01", "1996-10-01"], hidden_size=256,
                         gage_id=["01104430", "01175500", "01302000", "01302020", "01399670", "01443900", "02187500",
                                  "02277000", "02292900"],
                         gage_id_screen={"DOR": 0.1})
        var_o = {"RES_DOR_HIST": {"bins": 50}}
        self.args7 = cmd(sub="test/exp5", download=0, model_name="DapengCNNLSTM", opt="Adadelta", rs=1234,
                         model_param={"nx": len(var_c) + 7,
                                      "ny": 1,
                                      "nobs": 50,
                                      "hidden_size": 256,
                                      "n_kernel": (10, 5),
                                      "kernel_size": (3, 3),
                                      "stride": (2, 1),
                                      "dr": 0.5,
                                      "pool_opt": None,
                                      "cnn_dr": 0.5,
                                      "cat_first": False},
                         cache_write=1, scaler="DapengScaler", data_loader="KernelFlowDataModel", batch_size=5, rho=20,
                         weight_path=None, var_c=var_c, train_epoch=20, var_o=var_o,
                         loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                         test_period=["1995-10-01", "1996-10-01"],
                         gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                  "01055000", "01057000", "01170100"])

    def test_transfer_cropet_freeze_all(self):
        config_data = copy.deepcopy(self.config_data)
        update_cfg(config_data, self.args1)
        train_and_evaluate(config_data)

    def test_transfer_cropet_freeze_lstm(self):
        config_data = copy.deepcopy(self.config_data)
        update_cfg(config_data, self.args2)
        train_and_evaluate(config_data)

    def test_transfer_cropet_no_freeze(self):
        config_data = copy.deepcopy(self.config_data)
        update_cfg(config_data, self.args3)
        train_and_evaluate(config_data)

    def test_gages_dor_analysis(self):
        args = self.args4
        config_data = copy.deepcopy(self.config_data)
        gages = Gages(config_data["dataset_params"]["data_path"][1], False)
        if args.gage_id is not None or args.gage_id_file is not None:
            if args.gage_id_file is not None:
                gage_id_lst = pd.read_csv(args.gage_id_file, dtype={0: str}).iloc[:, 0].values
                usgs_ids = gage_id_lst.tolist()
            else:
                usgs_ids = args.gage_id
        else:
            usgs_ids = gages.read_object_ids()
        gage_id_screen = args.gage_id_screen
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
        train_and_evaluate(config_data)

    def test_lstm_kernel_dor(self):
        args = self.args5
        config_data = copy.deepcopy(self.config_data)
        gages = Gages(config_data["dataset_params"]["data_path"][1], False)
        if args.gage_id is not None or args.gage_id_file is not None:
            if args.gage_id_file is not None:
                gage_id_lst = pd.read_csv(args.gage_id_file, dtype={0: str}).iloc[:, 0].values
                usgs_ids = gage_id_lst.tolist()
            else:
                usgs_ids = args.gage_id
        else:
            usgs_ids = gages.read_object_ids()
        gage_id_screen = args.gage_id_screen
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
        train_and_evaluate(config_data)

    def test_lstm_kernel_dor_delta(self):
        args = self.args6
        config_data = copy.deepcopy(self.config_data)
        gages = Gages(config_data["dataset_params"]["data_path"][1], False)
        if args.gage_id is not None or args.gage_id_file is not None:
            if args.gage_id_file is not None:
                gage_id_lst = pd.read_csv(args.gage_id_file, dtype={0: str}).iloc[:, 0].values
                usgs_ids = gage_id_lst.tolist()
            else:
                usgs_ids = args.gage_id
        else:
            usgs_ids = gages.read_object_ids()
        gage_id_screen = args.gage_id_screen
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
        train_and_evaluate(config_data)

    def test_cnn_kernel_dor_hist(self):
        config_data = copy.deepcopy(self.config_data)
        update_cfg(config_data, self.args7)
        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = GagesPro(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)
        train_and_evaluate(config_data)

    def test_scaling_data(self):
        scaled_src, _ = self.model.test_data[0]
        data_unscaled = self.model.test_data.original_df.iloc[0:20]["cfs"].values
        inverse_scale = self.model.test_data.inverse_scale(scaled_src[:, 0])
        self.assertAlmostEqual(inverse_scale.numpy()[0], data_unscaled[0])
        self.assertAlmostEqual(inverse_scale.numpy()[9], data_unscaled[9])

    def test_compute_loss_no_scaling(self):
        exam = torch.Tensor([4.0]).repeat(2, 20, 5)
        exam2 = torch.Tensor([1.0]).repeat(2, 20, 5)
        exam11 = torch.Tensor([4.0]).repeat(2, 20)
        exam1 = torch.Tensor([1.0]).repeat(2, 20)
        d = DilateLoss()
        compute_loss(exam11, exam1, torch.rand(1, 20), d, None)
        # compute_loss(exam, exam2, torch.rand(2, 20), DilateLoss(), None)
        result = compute_loss(exam, exam2, torch.rand(2, 20), torch.nn.MSELoss(), None)
        self.assertEqual(float(result), 9.0)


if __name__ == '__main__':
    unittest.main()
