import unittest
import os
import numpy as np
import pandas as pd

import definitions
from data.cache.cache_factory import cache_dataset
from data.config import default_config_file, cmd, update_cfg
from data.pro.data_gages_pro import GagesPro
from utils.hydro_utils import unserialize_json_ordered


class AddForcingTests(unittest.TestCase):
    def setUp(self):
        # These are historical tests.
        # Firstly, choose the dammed basins with irrigation as the main purpose of reservoirs
        gages_dir = [os.path.join(definitions.DATASET_DIR, "gages_pro"),
                     os.path.join(definitions.DATASET_DIR, "gages"),
                     os.path.join(definitions.DATASET_DIR, "nid"),
                     os.path.join(definitions.DATASET_DIR, "gridmet")]
        dataset_name = "GAGES_PRO"
        self.config_data = default_config_file(gages_dir, dataset_name)
        # these attrs are not directly in GAGES-II; they need to be produced
        project_name = "gages_gridmet/exp1"
        attr_basin = ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'SLOPE_PCT']
        attr_landcover = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'BARRENNLCD06',
                          'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
        attr_soil = ['AWCAVE', 'PERMAVE', 'RFACT', 'ROCKDEPAVE']
        attr_geol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT']
        attr_hydro = ['STREAMS_KM_SQ_KM']
        attr_hydro_mod_dams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_DIS_NEAREST_MAJ_DAM']
        attr_hydro_mod_other = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'FRESHW_WITHDRAWAL', 'POWER_SUM_MW']
        attr_pop_infrastr = ['PDEN_2000_BLOCK', 'ROADS_KM_SQ_KM', 'IMPNLCD06']
        # attr_dam_related = ['DOR', 'DAM_MAIN_PURPOSE', 'DIVERSION', "DAM_GAGE_DIS_VAR", "DAM_STORAGE_STD"]
        var_c = attr_basin + attr_landcover + attr_soil + attr_geol + attr_hydro + attr_hydro_mod_dams + attr_hydro_mod_other + attr_pop_infrastr

        main_purpose_file = os.path.join(gages_dir[0], "dam_main_purpose_dict.json")
        all_sites_purposes_dict = unserialize_json_ordered(main_purpose_file)
        all_sites_purposes = pd.Series(all_sites_purposes_dict)
        include_irr = all_sites_purposes.apply(lambda x: "I" in x)
        irri_basins_id = all_sites_purposes[include_irr == True].index.tolist()

        gage_3557id_file = os.path.join(definitions.ROOT_DIR, "example", "3557basins_ID_NSE_DOR.csv")
        gage_id_lst = pd.read_csv(gage_3557id_file, dtype={0: str}).iloc[:, 0].values
        irri_basins_id = np.intersect1d(irri_basins_id, gage_id_lst)
        df = pd.DataFrame({"GAUGE_ID": irri_basins_id})
        df.to_csv(os.path.join(definitions.ROOT_DIR, "example", "328irrigation_gage_id.csv"), index=None)

        var_t_type = ["gridmet"]
        var_t = ["pr", "rmin", "srad", "tmmn", "tmmx", "vs"]
        # var_t = ["pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto"]
        # var_t = ["pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto", "cet"]

        self.args = cmd(sub=project_name, download=0, model_name="KuaiLSTM", opt="Adadelta", rs=1234, cache_write=1,
                        scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                        weight_path=None, var_c=var_c, n_feature=len(var_c) + len(var_t), train_epoch=2,
                        loss_func="RMSESum", train_period=["2008-01-01", "2013-01-01"],
                        test_period=["2013-01-01", "2018-01-01"], hidden_size=256,
                        var_t_type=var_t_type, var_t=var_t, gage_id=irri_basins_id.tolist())

    def test_gages_train_evaluate(self):
        config_data = self.config_data
        args = self.args
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = GagesPro(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        print("All processes are finished!")


if __name__ == "__main__":
    unittest.main()
