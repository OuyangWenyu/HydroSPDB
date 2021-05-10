import definitions
import unittest
import os

from data.cache.cache_factory import cache_dataset
from data.config import cmd, default_config_file, update_cfg
from data.data_gages import Gages
from data.pro.data_gages_pro import GagesPro
from hydroDL.trainer import train_and_evaluate


class TLTests(unittest.TestCase):
    def setUp(self):
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

        batch_size = 5
        rho = 20
        hidden_size = 256

        gages_dir = os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages")
        dataset1_name = "GAGES"
        self.config1_data = default_config_file(gages_dir, dataset1_name)
        project1_name = "gages/exp1"
        var_c1 = attr_basin + attr_landcover + attr_soil + attr_geol + attr_hydro + attr_hydro_mod_dams + attr_hydro_mod_other + attr_pop_infrastr
        self.args1 = cmd(sub=project1_name, download=0, model_name="KuaiLSTM", opt="Adadelta", rs=1234, cache_write=1,
                         scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=batch_size, rho=rho,
                         weight_path=None, var_c=var_c1, n_feature=len(var_c1) + 7, train_epoch=20,
                         loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                         test_period=["1995-10-01", "1996-10-01"], hidden_size=hidden_size,
                         gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                  "01055000", "01057000", "01170100"], )

        gages_pro_dir = [os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages_pro"),
                         gages_dir,
                         os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "nid"),
                         os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gridmet")]
        dataset2_name = "GAGES_PRO"
        self.config2_data = default_config_file(gages_pro_dir, dataset2_name)
        # these attrs are not directly in GAGES-II; they need to be produced
        project2_name = "gages_add_attr/exp1"
        var_c2 = attr_basin + attr_landcover + attr_soil + attr_geol + attr_hydro + attr_hydro_mod_dams + attr_hydro_mod_other + attr_pop_infrastr + attr_dam_related
        self.args2 = cmd(sub=project2_name, download=0, model_name="KaiTlLSTM", opt="Adadelta", rs=1234, cache_write=1,
                         scaler="DapengScaler", data_loader="StreamflowDataModel",
                         weight_path_add={
                             "freeze_params": ["linearIn.bias", "linearIn.weight", "linearOut.bias", "linearOut.weight",
                                               "lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]},
                         weight_path="/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp1/10_April_202111_21AM_model.pth",
                         var_c=var_c2, n_feature=len(var_c2) + 7, train_epoch=20, continue_train=True,
                         loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                         test_period=["1995-10-01", "1996-10-01"],
                         gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01057000", "01170100"],
                         model_param={"seq_length": rho,
                                      "n_time_series": len(var_c2) + 7,
                                      "input_seq_len": len(var_c1) + 7,
                                      "output_seq_len": 1,
                                      "hidden_states": hidden_size,
                                      "num_layers": 1,
                                      "bias": True,
                                      "batch_size": batch_size,
                                      "probabilistic": False})

    def test_generate_source_model(self):
        config_data = self.config1_data
        args = self.args1
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = Gages(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)
        print("All processes are finished!")

    def test_transfer_model(self):
        config_data = self.config2_data
        args = self.args2
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = GagesPro(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)
        print("All processes are finished!")


if __name__ == '__main__':
    unittest.main()
