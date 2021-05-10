import os
from unittest import TestCase

import definitions
from data.config import default_config_file, cmd, update_cfg
from data.data_gages import Gages
from data.pro.select_gages_ids import usgs_screen_streamflow, dor_reservoirs_chosen


class Test(TestCase):
    def setUp(self):
        gages_dir = os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages")
        dataset_name = "GAGES"
        config_data = default_config_file(gages_dir, dataset_name)
        project_name = "gages/exp1"
        attrBasin = ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'SLOPE_PCT']
        attrLandcover = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'BARRENNLCD06',
                         'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
        attrSoil = ['AWCAVE', 'PERMAVE', 'RFACT', 'ROCKDEPAVE']
        attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT']
        attrHydro = ['STREAMS_KM_SQ_KM']
        attrHydroModDams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_DIS_NEAREST_MAJ_DAM']
        attrHydroModOther = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'FRESHW_WITHDRAWAL', 'POWER_SUM_MW']
        attrPopInfrastr = ['PDEN_2000_BLOCK', 'ROADS_KM_SQ_KM', 'IMPNLCD06']
        varC = attrBasin + attrLandcover + attrSoil + attrGeol + attrHydro + attrHydroModDams + attrHydroModOther + attrPopInfrastr
        args = cmd(sub=project_name, download=0, model_name="KuaiLSTM", opt="Adadelta", rs=1234, cache_write=1,
                   scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                   var_c=varC, n_feature=len(varC) + 7, train_epoch=20,
                   loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                   test_period=["1995-10-01", "1996-10-01"], hidden_size=256,
                   gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                            "01055000", "01057000", "01170100"], )
        update_cfg(config_data, args)
        self.config = config_data
        self.gages = Gages(gages_dir)

    def test_usgs_screen_streamflow(self):
        gages = self.gages
        config_data = self.config
        flow_screen_param = {'missing_data_ratio': 0, 'zero_value_ratio': 1}
        selected_ids = usgs_screen_streamflow(gages, config_data["dataset_params"]["object_ids"],
                                              config_data["dataset_params"]["t_range_train"], **flow_screen_param)
        self.assertEqual(len(selected_ids), 10)

    def test_dor_reservoirs_chosen(self):
        gages = self.gages
        config_data = self.config
        selected_ids = dor_reservoirs_chosen(gages, config_data["dataset_params"]["object_ids"], dor_chosen=-0.1)
        self.assertEqual(len(selected_ids), 10)
