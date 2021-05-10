import unittest
import os
from functools import reduce

import definitions
from data.cache.cache_factory import cache_dataset
from data.config import default_config_file, cmd, update_cfg
from data.pro.data_gages_pro import GagesPro
from hydroDL.trainer import train_and_evaluate


class CnnKernelTest(unittest.TestCase):
    def setUp(self):
        gages_dir = [os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages_pro"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "nid"),
                     os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gridmet")]
        dataset_name = "GAGES_PRO"
        self.config_data = default_config_file(gages_dir, dataset_name)
        project_name = "gages_add_attr/exp2"
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
        var_o = {"RES_STOR_HIST": {"bins": 50}}
        self.args = cmd(sub=project_name, download=0, model_name="DapengCNNLSTM", opt="Adadelta", rs=1234,
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
                        weight_path=None, var_c=var_c, n_feature=len(var_c) + 7, train_epoch=20, var_o=var_o,
                        loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                        test_period=["1995-10-01", "1996-10-01"], hidden_size=64,
                        gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                 "01055000", "01057000", "01170100"])

    def test_cnn_kernel(self):
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
