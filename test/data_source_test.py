import unittest
import numpy as np
import os

import definitions
from data.config import default_config_file, cmd, update_cfg
from data.data_gages import Gages
from data.pro.data_gages_pro import get_dor_values, get_diversion, get_dam_storage_std, get_dam_dis_var, GagesPro, \
    get_dam_main_purpose, get_reservoir_dor_hist


class TestPreprocessCode(unittest.TestCase):
    def setUp(self):
        gages_dir = os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages")
        project_name = r"gages/exp1"
        dataset_name = "GAGES"
        config_data = default_config_file(gages_dir, dataset_name)
        weight_path = None
        # weight_path = os.path.join(config_data["dataset_params"]["test_path"], "26_March_202109_27AM_model.pth")
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
        # self.usgs_id = ["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
        #                 "01055000", "01057000", "01170100"]
        self.usgs_id = ["01013500", "01016500", "01022500", "01057000"]

        var_t_type = ["gridmet"]
        var_t = ["pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto", "etr"]
        args = cmd(sub=project_name, download=False, model_name="KuaiLSTM", opt="Adadelta", rs=1234, cache_read=0,
                   scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                   weight_path=weight_path, var_c=varC, n_feature=len(varC) + len(var_t), gage_id=self.usgs_id,
                   var_t_type=var_t_type, var_t=var_t, train_period=["2009-01-01", "2010-01-01"])
        # gage_id_file="/mnt/data/owen411/code/hydro-spdb-dl/example/3557basins_ID_NSE_DOR.csv",)
        update_cfg(config_data, args)
        self.config = config_data
        self.gages = Gages(gages_dir)
        nid_dir = os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "nid")
        gridmet_dir = os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gridmet")
        gages_pro_dir = os.path.join("/".join(definitions.ROOT_DIR.split("/")[0:-2]), "data", "gages_pro")
        data_dir_lst = [gages_pro_dir, gages_dir, nid_dir, gridmet_dir]
        self.gages_pro = GagesPro(data_dir_lst)

    def test_read_streamflow(self):
        print("read streamflow data")
        gages_pro = self.gages_pro
        config_data = self.config
        streamflow = gages_pro.read_target_cols(object_ids=config_data["dataset_params"]["object_ids"],
                                                t_range_list=config_data["dataset_params"]["t_range_train"],
                                                target_cols=config_data["dataset_params"]["target_cols"])
        self.assertEqual(streamflow[3, 6], 102)

    def test_read_cropet(self):
        print("read cropet data")
        gages_pro = self.gages_pro
        config_data = self.config
        relevant_cols = ["pr", "cet", "eto"]
        forcings = gages_pro.read_relevant_cols(object_ids=config_data["dataset_params"]["object_ids"],
                                                t_range_list=config_data["dataset_params"]["t_range_train"],
                                                relevant_cols=relevant_cols,
                                                forcing_type=config_data["dataset_params"]["relevant_types"][0])
        self.assertEqual(forcings[3, 6, 2], 0.56)

    def test_read_new_forcing_data(self):
        print("read new forcing data")
        gages_pro = self.gages_pro
        config_data = self.config
        forcings = gages_pro.read_relevant_cols(object_ids=config_data["dataset_params"]["object_ids"],
                                                t_range_list=config_data["dataset_params"]["t_range_train"],
                                                relevant_cols=config_data["dataset_params"]["relevant_cols"],
                                                forcing_type=config_data["dataset_params"]["relevant_types"][0])
        self.assertEqual(forcings[3, 6, 6], 0.56)

    def test_read_new_attr_date(self):
        print("test if compatible with previous version")
        gages_pro = self.gages_pro
        config_data = self.config
        attrs = gages_pro.read_constant_cols(object_ids=config_data["dataset_params"]["object_ids"],
                                             constant_cols=config_data["dataset_params"]["constant_cols"])
        self.assertEqual(attrs[3, 6], 1.58)

    def test_get_dor_values(self):
        gages = self.gages
        usgs_id = self.usgs_id
        dors = get_dor_values(gages, usgs_id)
        print(dors)
        dors_format = np.array([format(i, '.3g') for i in dors])
        assert (dors_format == np.array(
            [format(770 / 604100, '.3g'), format(790 / 583600, '.3g'), format(14940 / 760000, '.3g'),
             format(53890 / 717000, '.3g')])).all()

    def test_get_diversion(self):
        gages = self.gages
        usgs_id = self.usgs_id
        diversions = get_diversion(gages, usgs_id)
        print(diversions)
        assert diversions.tolist() == [False, False, False, False]

    def test_get_dam_dis_var(self):
        gages_pro = self.gages_pro
        usgs_id = self.usgs_id
        dis_var = get_dam_dis_var(gages_pro, usgs_id)
        print(dis_var)
        dis_var = np.array([format(i, '.3g') for i in dis_var if i == i])
        assert (dis_var == np.array([format(0.0, '.3g'), format(64.3584968424885, '.3g'),
                                     format(14.584108603093854, '.3g')])).all()
        usgs_id = ["01013500", "01022500", "01057000"]
        dis_var = get_dam_dis_var(gages_pro, usgs_id)
        dis_var = np.array([format(i, '.3g') for i in dis_var])
        assert (dis_var == np.array([format(0.0, '.3g'), format(64.3584968424885, '.3g'),
                                     format(14.584108603093854, '.3g')])).all()

    def test_get_dam_main_purpose(self):
        gages = self.gages_pro
        usgs_id = self.usgs_id
        gage_main_dam_purpose = get_dam_main_purpose(gages, usgs_id)
        print(gage_main_dam_purpose)
        gage_main_dam_purpose = np.array([i for i in gage_main_dam_purpose if i is not None])
        print(gage_main_dam_purpose)
        assert (gage_main_dam_purpose == np.array(['F', 'O', 'O'])).all()

    def test_get_dam_storage_std(self):
        gages_pro = self.gages_pro
        usgs_id = self.usgs_id
        dam_stor_std = get_dam_storage_std(gages_pro, usgs_id)
        print(dam_stor_std)
        dam_stor_std = np.array([format(i, '.3g') for i in dam_stor_std if i == i])
        assert (dam_stor_std == np.array([format(0.0, '.3g'), format(8.326702524014614, '.3g'),
                                          format(3.8501476017100584, '.3g')])).all()

    def test_get_reservoir_dor_hist(self):
        gages_pro = self.gages_pro
        usgs_id = self.usgs_id
        dam_dor_hist = get_reservoir_dor_hist(gages_pro, usgs_id)
        print(dam_dor_hist)
        self.assertEqual(dam_dor_hist[0, 0], 1)


if __name__ == '__main__':
    unittest.main()
