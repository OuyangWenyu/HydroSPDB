import os

import definitions
from data.cache.cache_factory import cache_dataset
from data.config import default_config_file, cmd, update_cfg
from data.data_camels import Camels
from hydroDL.basic.lstm_vanilla import LSTMForecast, CudaLSTM
import unittest
import torch

from hydroDL.trainer import train_and_evaluate


class TestLstm(unittest.TestCase):
    def setUp(self):
        lstm1 = LSTMForecast(seq_length=10,
                             n_time_series=5,
                             output_seq_len=1,
                             hidden_states=20,
                             num_layers=1,
                             bias=True,
                             batch_size=15,
                             probabilistic=False)
        self.lstm1 = lstm1.to(lstm1.device)
        lstm2 = LSTMForecast(seq_length=10,
                             n_time_series=5,
                             output_seq_len=1,
                             hidden_states=20,
                             num_layers=1,
                             bias=True,
                             batch_size=15,
                             probabilistic=False,
                             mode="NtoN")
        self.lstm2 = lstm2.to(lstm2.device)
        lstm3 = CudaLSTM(n_time_series=5,
                         output_seq_len=1,
                         hidden_size=20)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm3 = lstm3.to(device)
        camels_dir = os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "camels")
        dataset_name = "CAMELS"
        self.config_data = default_config_file(camels_dir, dataset_name)
        weight_path = None
        self.args1 = cmd(sub="test/exp8", download=0, model_name="LSTM", cache_write=1,
                         model_param={"seq_length": 20,
                                      "n_time_series": 23,
                                      "output_seq_len": 1,
                                      "hidden_states": 64,
                                      "num_layers": 1,
                                      "bias": True,
                                      "batch_size": 5,
                                      "probabilistic": False,
                                      "mode": "NtoN"},
                         gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                  "01055000", "01057000", "01170100"], batch_size=5,
                         var_t=["dayl", "prcp", 'srad', 'tmax', 'tmin', 'vp'],
                         data_loader="StreamflowDataset", scaler="DapengScaler",
                         weight_path=weight_path, train_epoch=2)
        self.args2 = cmd(sub="test/exp9", download=0, model_name="LSTM", cache_write=1,
                         model_param={"seq_length": 20,
                                      "n_time_series": 23,
                                      "output_seq_len": 1,
                                      "hidden_states": 64,
                                      "num_layers": 1,
                                      "bias": True,
                                      "batch_size": 5,
                                      "probabilistic": False,
                                      "mode": "Nto1"},
                         gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                  "01055000", "01057000", "01170100"], batch_size=5,
                         var_t=["dayl", "prcp", 'srad', 'tmax', 'tmin', 'vp'],
                         data_loader="SingleflowDataset", scaler="DapengScaler",
                         weight_path=weight_path, train_epoch=2)
        self.args3 = cmd(sub="test/exp10", download=0, model_name="FreddyLSTM", cache_write=1,
                         model_param={"n_time_series": 23,
                                      "output_seq_len": 1,
                                      "hidden_size": 64},
                         gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                  "01055000", "01057000", "01170100"], batch_size=5, rho=10,
                         var_t=["dayl", "prcp", 'srad', 'tmax', 'tmin', 'vp'],
                         data_loader="SingleflowDataset", scaler="DapengScaler",
                         weight_path=weight_path, train_epoch=2)

        self.args4 = cmd(sub="test/exp10", download=0, model_name="FreddyLSTM", cache_write=1,
                         model_param={"n_time_series": 23,
                                      "output_seq_len": 1,
                                      "hidden_size": 64},
                         gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                  "01055000", "01057000", "01170100"], batch_size=5, rho=10,
                         var_t=["dayl", "prcp", 'srad', 'tmax', 'tmin', 'vp'],
                         data_loader="SingleflowDataset", scaler="DapengScaler",
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/test/test_models/22_April_202111_17AM_model.pth",
                         continue_train=0,
                         train_epoch=2)

    def test_freddy_lstm(self):
        # batch, time_sequence, feature_size
        a = torch.rand(15, 10, 5).to(self.lstm1.device)
        r = self.lstm3(a)
        # N-to-1 model
        self.assertEqual(len(r.shape), 2)
        self.assertIsInstance(r, torch.Tensor)

    def test_evaluate_freddy_lstm(self):
        config_data = self.config_data
        args = self.args4
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = Camels(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)

    def test_train_evaluate_freddy_lstm(self):
        config_data = self.config_data
        args = self.args3
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = Camels(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)

    def test_vanilla_lstm(self):
        a = torch.rand(15, 10, 5).to(self.lstm1.device)

        r = self.lstm1(a)
        # N-to-1 model
        self.assertEqual(len(r.shape), 2)
        self.assertIsInstance(r, torch.Tensor)

    def test_vanilla_lstm_n2n_mode(self):
        a = torch.rand(15, 10, 5).to(self.lstm2.device)

        r = self.lstm2(a)
        # N-to-N model
        self.assertEqual(len(r.shape), 3)
        self.assertIsInstance(r, torch.Tensor)

    def test_train_evaluate(self):
        config_data = self.config_data
        args = self.args1
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = Camels(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)

    def test_train_evaluate_n21(self):
        config_data = self.config_data
        args = self.args2
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = Camels(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)
