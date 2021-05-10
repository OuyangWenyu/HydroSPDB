import torch
import unittest
import os
import tempfile

import definitions
from data.cache.cache_factory import cache_dataset
from data.config import default_config_file, cmd, update_cfg
from data.data_camels import Camels
from data.data_station import make_data
from hydroDL.da_rnn.model import DARNN
from hydroDL.da_rnn.train_da import da_rnn, train
from hydroDL.trainer import train_and_evaluate
from utils.hydro_utils import device


class TestDARNN(unittest.TestCase):
    def setUp(self):
        self.preprocessed_data = make_data(os.path.join(os.path.dirname(__file__), "test_init", "keag_small.csv"),
                                           ["cfs"], 72)
        camels_dir = os.path.join(definitions.DATASET_DIR, "camels")
        dataset_name = "CAMELS"
        project_name = "test/exp7"
        self.config_data = default_config_file(camels_dir, dataset_name)
        weight_path = None
        self.args = cmd(sub=project_name, download=0, model_name="DARNN", cache_write=1,
                        model_param={"n_time_series": 23,
                                     "hidden_size_encoder": 64,
                                     "forecast_history": 20,
                                     "decoder_hidden_size": 64,
                                     "out_feats": 1,
                                     "dropout": .01,
                                     "meta_data": False,
                                     "gru_lstm": True,
                                     "probabilistic": False,
                                     "final_act": None,
                                     "data_integration": False},
                        gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                 "01055000", "01057000", "01170100"], batch_size=5,
                        var_t=["dayl", "prcp", 'srad', 'tmax', 'tmin', 'vp'],
                        data_loader="SingleflowDataset", scaler="DapengScaler",
                        weight_path=weight_path, train_epoch=2)

    def test_new_darnn(self):
        darnn = DARNN(n_time_series=3,
                      hidden_size_encoder=64,
                      forecast_history=100,
                      decoder_hidden_size=64,
                      out_feats=1,
                      dropout=.01,
                      meta_data=False,
                      gru_lstm=True,
                      probabilistic=False,
                      final_act=None).to(device)
        # the data has no physical meaning, just for a model test
        a = torch.Tensor(self.preprocessed_data.feats[:2000, :].reshape(20, -1, 3)).to(device)
        r = darnn(a)
        self.assertEqual(r.shape[-1], 1)

    def test_new_darnn_no_DI(self):
        darnn = DARNN(n_time_series=3,
                      hidden_size_encoder=64,
                      forecast_history=100,
                      decoder_hidden_size=64,
                      out_feats=1,
                      dropout=.01,
                      meta_data=False,
                      gru_lstm=True,
                      probabilistic=False,
                      final_act=None, data_integration=False).to(device)
        # the data has no physical meaning, just for a model test
        a = torch.Tensor(self.preprocessed_data.feats[:2000, :].reshape(20, -1, 3)).to(device)
        r = darnn(a)
        print(r)
        self.assertEqual(r.shape[-1], 1)

    def test_train_evaluate(self):
        config_data = self.config_data
        args = self.args
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = Camels(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)

    def test_train_model(self):
        with tempfile.TemporaryDirectory() as param_directory:
            config, da_network = da_rnn(self.preprocessed_data, 1, 64,
                                        param_output_path=param_directory)
            loss_results, model = train(da_network, self.preprocessed_data,
                                        config, n_epochs=1, tensorboard=True)
            self.assertTrue(model)

    def test_tf_data(self):
        dirname = os.path.dirname(__file__)
        # Test that Tensorboard directory was indeed created
        self.assertTrue(os.listdir(os.path.join(dirname)))

    def test_create_model(self):
        with tempfile.TemporaryDirectory() as param_directory:
            config, dnn_network = da_rnn(self.preprocessed_data, 1, 64,
                                         param_output_path=param_directory)
            self.assertNotEqual(config.batch_size, 20)
            self.assertIsNotNone(dnn_network)

    def test_resume_ckpt(self):
        """ This test is dependent on test_train_model succeding"""
        config, da = da_rnn(self.preprocessed_data, 1, 64)
        with tempfile.TemporaryDirectory() as checkpoint:
            torch.save(da.encoder.state_dict(), os.path.join(checkpoint, "encoder.pth"))
            torch.save(da.decoder.state_dict(), os.path.join(checkpoint, "decoder.pth"))
            config, dnn_network = da_rnn(self.preprocessed_data, 1, 64, save_path=checkpoint)
            self.assertTrue(dnn_network)


if __name__ == '__main__':
    unittest.main()
