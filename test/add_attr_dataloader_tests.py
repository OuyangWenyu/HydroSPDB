import unittest
import os

import definitions
from data.cache.cache_factory import cache_dataset
from data.config import default_config_file, cmd, update_cfg
from data.pro.data_gages_pro import GagesPro
from hydroDL.trainer import train_and_evaluate


class AddNewDataLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        gages_dir = [os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages_pro"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "nid"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gridmet")]
        dataset_name = "GAGES_PRO"
        self.config_data = default_config_file(gages_dir, dataset_name)
        # these attrs are not directly in GAGES-II; they need to be produced
        project_name = "gages_add_attr/exp1"
        attr_basin = ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'SLOPE_PCT']
        attr_landcover = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'BARRENNLCD06',
                          'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
        attr_soil = ['AWCAVE', 'PERMAVE', 'RFACT', 'ROCKDEPAVE']
        attr_geol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT']
        attr_hydro = ['STREAMS_KM_SQ_KM']
        attr_hydro_mod_dams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_DIS_NEAREST_MAJ_DAM']
        attr_hydro_mod_other = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'FRESHW_WITHDRAWAL', 'POWER_SUM_MW']
        attr_pop_infrastr = ['PDEN_2000_BLOCK', 'ROADS_KM_SQ_KM', 'IMPNLCD06']
        attr_dam_related = ['DOR', 'DAM_MAIN_PURPOSE', 'DIVERSION', "DAM_GAGE_DIS_VAR", "DAM_STORAGE_STD"]
        var_c = attr_basin + attr_landcover + attr_soil + attr_geol + attr_hydro + attr_hydro_mod_dams + attr_hydro_mod_other + attr_pop_infrastr + attr_dam_related
        self.args = cmd(sub=project_name, download=0, model_name="KuaiLSTM", opt="Adadelta", rs=1234, cache_write=1,
                        scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                        weight_path=None, var_c=var_c, n_feature=len(var_c) + 7, train_epoch=20,
                        loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                        test_period=["1995-10-01", "1996-10-01"], hidden_size=256,
                        gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                 "01055000", "01057000", "01170100"], )
        self.args1 = cmd(sub="gages/exp15", download=0, model_name="KaiTlLSTM", opt="Adadelta", rs=1234,
                         cache_read=1,
                         cache_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/cache-2909sites-19900101_20000101_20000101_20100101-35attr-7forcing",
                         scaler="DapengScaler", data_loader="StreamflowDataModel",
                         model_param={"seq_length": 365, "n_time_series": 42, "input_seq_len": 37, "output_seq_len": 1,
                                      "hidden_states": 256, "num_layers": 1, "bias": True, "batch_size": 100,
                                      "probabilistic": False},
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp14/model_Ep250.pth",
                         continue_train=0, var_c=var_c, n_feature=42, train_epoch=250,
                         loss_func="RMSESum", train_period=["1990-01-01", "2000-01-01"],
                         test_period=["2000-01-01", "2010-01-01"], hidden_size=256,
                         gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/2909basins_NSE.csv")
        self.args2 = cmd(sub="gages/exp15", download=0, model_name="KaiTlLSTM", opt="Adadelta", rs=1234,
                         cache_read=1,
                         cache_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/cache-2909sites-19900101_20000101_20000101_20100101-35attr-7forcing",
                         scaler="DapengScaler", data_loader="StreamflowDataModel",
                         model_param={"seq_length": 365, "n_time_series": 42, "input_seq_len": 37, "output_seq_len": 1,
                                      "hidden_states": 256, "num_layers": 1, "bias": True, "batch_size": 100,
                                      "probabilistic": False},
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp14/model_Ep200.pth",
                         continue_train=0, var_c=var_c, n_feature=42, train_epoch=200,
                         loss_func="RMSESum", train_period=["1990-01-01", "2000-01-01"],
                         test_period=["2000-01-01", "2010-01-01"], hidden_size=256,
                         gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/2909basins_NSE.csv")
        self.args3 = cmd(sub="gages/exp15", download=0, model_name="KaiTlLSTM", opt="Adadelta", rs=1234,
                         cache_read=1,
                         cache_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/cache-2909sites-19900101_20000101_20000101_20100101-35attr-7forcing",
                         scaler="DapengScaler", data_loader="StreamflowDataModel",
                         model_param={"seq_length": 365, "n_time_series": 42, "input_seq_len": 37, "output_seq_len": 1,
                                      "hidden_states": 256, "num_layers": 1, "bias": True, "batch_size": 100,
                                      "probabilistic": False},
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp14/model_Ep150.pth",
                         continue_train=0, var_c=var_c, n_feature=42, train_epoch=150,
                         loss_func="RMSESum", train_period=["1990-01-01", "2000-01-01"],
                         test_period=["2000-01-01", "2010-01-01"], hidden_size=256,
                         gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/2909basins_NSE.csv")
        self.args4 = cmd(sub="gages/exp15", download=0, model_name="KaiTlLSTM", opt="Adadelta", rs=1234,
                         cache_read=1,
                         cache_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/cache-2909sites-19900101_20000101_20000101_20100101-35attr-7forcing",
                         scaler="DapengScaler", data_loader="StreamflowDataModel",
                         model_param={"seq_length": 365, "n_time_series": 42, "input_seq_len": 37, "output_seq_len": 1,
                                      "hidden_states": 256, "num_layers": 1, "bias": True, "batch_size": 100,
                                      "probabilistic": False},
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp14/model_Ep100.pth",
                         continue_train=0, var_c=var_c, n_feature=42, train_epoch=100,
                         loss_func="RMSESum", train_period=["1990-01-01", "2000-01-01"],
                         test_period=["2000-01-01", "2010-01-01"], hidden_size=256,
                         gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/2909basins_NSE.csv")
        self.args5 = cmd(sub="gages/exp15", download=0, model_name="KaiTlLSTM", opt="Adadelta", rs=1234,
                         cache_read=1,
                         cache_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/cache-2909sites-19900101_20000101_20000101_20100101-35attr-7forcing",
                         scaler="DapengScaler", data_loader="StreamflowDataModel",
                         model_param={"seq_length": 365, "n_time_series": 42, "input_seq_len": 37, "output_seq_len": 1,
                                      "hidden_states": 256, "num_layers": 1, "bias": True, "batch_size": 100,
                                      "probabilistic": False},
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp14/model_Ep50.pth",
                         continue_train=0, var_c=var_c, n_feature=42, train_epoch=50,
                         loss_func="RMSESum", train_period=["1990-01-01", "2000-01-01"],
                         test_period=["2000-01-01", "2010-01-01"], hidden_size=256,
                         gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/2909basins_NSE.csv")

    def test_train_gages_test(self):
        config_data = self.config_data
        args = self.args4
        update_cfg(config_data, args)

        train_and_evaluate(config_data)
        print("All processes are finished!")

    def test_train_gages(self):
        config_data = self.config_data
        args = self.args
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = GagesPro(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)
        print("All processes are finished!")


if __name__ == '__main__':
    unittest.main()
