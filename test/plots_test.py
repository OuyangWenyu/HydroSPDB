import copy
import os
import unittest
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import numpy as np

import definitions
from data.config import default_config_file, cmd, update_cfg
from explore.stat import ecdf
from hydroDL.trainer import stat_ensemble_result, stat_result
from visual.plot_functions import calculate_confidence_intervals, plot_df_test_with_confidence_interval
from visual.plot_stat import plot_ecdfs_matplot


class PlotFunctionsTest(unittest.TestCase):
    """ Tests the plot functions
    """
    df_test = pd.DataFrame({
        'preds': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        'target_col': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    })
    df_preds = pd.DataFrame({
        0: [-1.0, -2.0, -1.0, 0.0, -1.0, 6.0],
        1: [1.0, 2.0, 4.0, 3.0, 2.0, 9.0]
    })
    df_preds_empty = pd.DataFrame(index=[0, 1, 2, 3, 4, 5])

    def setUp(self) -> None:
        gages_dir = [os.path.join(definitions.DATASET_DIR, "gages_pro"),
                     os.path.join(definitions.DATASET_DIR, "gages"),
                     os.path.join(definitions.DATASET_DIR, "nid"),
                     os.path.join(definitions.DATASET_DIR, "gridmet")]
        dataset_name = "GAGES_PRO"
        config_data = default_config_file(gages_dir, dataset_name)
        # these attrs are not directly in GAGES-II; they need to be produced
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

        irrigation328_gage_id = os.path.join(definitions.ROOT_DIR, "example", "328irrigation_gage_id.csv")
        self.irri_basins_id = pd.read_csv(irrigation328_gage_id, dtype={0: str}).iloc[:, 0].values
        gage_3557id_file = os.path.join(definitions.ROOT_DIR, "example", "3557basins_ID_NSE_DOR.csv")
        self.basins_id = pd.read_csv(gage_3557id_file, dtype={0: str}).iloc[:, 0].values

        var_t_type = ["gridmet"]
        var_t = ["pr", "rmin", "srad", "tmmn", "tmmx", "vs"]

        exps_lst = ["gages_gridmet/exp1"]
        cfgs = []
        for project_name in exps_lst:
            new_cfg = copy.deepcopy(config_data)
            args = cmd(sub=project_name, download=0, model_name="KuaiLSTM", opt="Adadelta", rs=1234, cache_write=1,
                       scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                       weight_path=None, var_c=var_c, n_feature=len(var_c) + len(var_t), train_epoch=2,
                       loss_func="RMSESum", train_period=["2008-01-01", "2013-01-01"],
                       test_period=["2013-01-01", "2018-01-01"], hidden_size=256,
                       var_t_type=var_t_type, var_t=var_t, gage_id=self.irri_basins_id.tolist())
            update_cfg(new_cfg, args)
            cfgs.append(new_cfg)
        self.cfgs = cfgs

    def test_compare_projects(self):
        cfgs = self.cfgs
        test_epoch = cfgs[0]["training_params"]["epochs"]
        keys_nse = "NSE"

        inds_df1 = stat_ensemble_result([cfgs[0]["dataset_params"]["test_path"]], test_epoch)
        inds_df2 = stat_result(cfgs[0]["dataset_params"]["test_path"], test_epoch)
        inds_df3 = stat_ensemble_result([cfgs[0]["dataset_params"]["test_path"]], test_epoch)
        xs1 = []
        ys1 = []
        cases_exps_legends_together_1 = ["328_cet", "328_et", "328_no-et"]

        x1, y1 = ecdf(np.nan_to_num(inds_df1[keys_nse]))
        xs1.append(x1)
        ys1.append(y1)

        x2, y2 = ecdf(np.nan_to_num(inds_df2[keys_nse]))
        xs1.append(x2)
        ys1.append(y2)

        x3, y3 = ecdf(np.nan_to_num(inds_df3[keys_nse]))
        xs1.append(x3)
        ys1.append(y3)

        plot_ecdfs_matplot(xs1, ys1, cases_exps_legends_together_1,
                           colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
                           dash_lines=[False, False, False], x_str="NSE", y_str="CDF")
        plt.show()

    def test_calculate_confidence_intervals_df_preds_empty(self):
        ci_lower, ci_upper = 0.025, 0.975
        df_quantiles = calculate_confidence_intervals(
            self.df_preds_empty, self.df_test['preds'], ci_lower, ci_upper)
        self.assertTrue(df_quantiles[ci_lower].isna().all())
        self.assertTrue(df_quantiles[ci_upper].isna().all())

    def test_plot_df_test_with_confidence_interval(self):
        params = {'dataset_params': {'target_col': ['target_col']}}
        fig = plot_df_test_with_confidence_interval(self.df_test, self.df_preds, 0, params, "target_col", 95)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_df_test_with_confidence_interval_df_preds_empty(self):
        params = {'dataset_params': {'target_col': ['target_col']}}
        fig = plot_df_test_with_confidence_interval(
            self.df_test, self.df_preds_empty, 0, params, "target_col", 95)
        self.assertIsInstance(fig, go.Figure)


if __name__ == '__main__':
    unittest.main()
