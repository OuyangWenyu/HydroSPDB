import unittest
import pandas as pd
import copy
import os
import numpy as np
from data import GagesConfig
from data.gages_input_dataset import load_ensemble_result
from explore.stat import ecdf
import matplotlib.pyplot as plt
from utils import unserialize_json_ordered
from data.config import cfg
from visual.plot_stat import plot_ecdfs_matplot


class MyTestCase(unittest.TestCase):
    """data pre-process and post-process"""

    def setUp(self) -> None:
        nid_dir = os.path.join(cfg.NID.NID_DIR, "test")
        main_purpose_file = os.path.join(nid_dir, "dam_main_purpose_dict.json")
        all_sites_purposes_dict = unserialize_json_ordered(main_purpose_file)
        all_sites_purposes = pd.Series(all_sites_purposes_dict)
        include_irr = all_sites_purposes.apply(lambda x: "I" in x)
        self.irri_basins_id = all_sites_purposes[include_irr == True].index.tolist()

        dir_3557 = os.path.join(cfg.DATA_PATH, "quickdata", "conus-all_90-10_nan-0.0_00-1.0")
        timespace_file = os.path.join(dir_3557, "dictTimeSpace.json")
        all_sites_dict = unserialize_json_ordered(timespace_file)
        self.basins_id = all_sites_dict["sites_id"]

        self.t_range_train = ["2008-01-01", "2013-01-01"]
        self.t_range_test = ["2013-01-01", "2018-01-01"]

        exps_lst = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
        cfgs = []
        data_configs = []
        for exp in exps_lst:
            new_cfg = copy.deepcopy(cfg)

            new_cfg.SUBSET = "gridmet"
            new_cfg.SUB_EXP = exp
            new_cfg.TEMP_PATH = os.path.join(new_cfg.ROOT_DIR, 'temp', new_cfg.DATASET,
                                             new_cfg.SUBSET, new_cfg.SUB_EXP)
            new_cfg.OUT_PATH = os.path.join(new_cfg.ROOT_DIR, 'output', new_cfg.DATASET,
                                            new_cfg.SUBSET, new_cfg.SUB_EXP)

            new_cfg.MODEL.tRangeTrain = self.t_range_train
            new_cfg.MODEL.tRangeTest = self.t_range_test
            new_cfg.GAGES.streamflowScreenParams = {'missing_data_ratio': 1, 'zero_value_ratio': 1}
            new_cfg.CACHE.QUICK_DATA = False
            new_cfg.CACHE.GEN_QUICK_DATA = True
            config_data = GagesConfig(new_cfg)
            cfgs.append(new_cfg)
            data_configs.append(config_data)

        self.cfgs = cfgs
        self.data_configs = data_configs
        self.exp_lst1 = ["gridmet_exp1"]
        self.exp_lst2 = ["gridmet_exp2"]
        self.exp_lst3 = ["gridmet_exp3"]
        self.exp_lst4 = ["gridmet_exp4"]
        self.exp_lst5 = ["gridmet_exp5"]
        self.exp_lst6 = ["gridmet_exp6"]

    def test_compare_projects(self):
        cfgs = self.cfgs
        config_datas = self.data_configs

        keys_nse = "NSE"

        inds_df1 = load_ensemble_result(cfgs[0], self.exp_lst1, test_epoch=300)
        inds_df2 = load_ensemble_result(cfgs[1], self.exp_lst2, test_epoch=300)
        inds_df3 = load_ensemble_result(cfgs[2], self.exp_lst3, test_epoch=300)
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
        plt.savefig(os.path.join(config_datas[0].data_path["Out"], '328_cet_et_no-et.png'), dpi=600,
                    bbox_inches="tight")

        inds_df4 = load_ensemble_result(cfgs[3], self.exp_lst4, test_epoch=300)
        inds_df5 = load_ensemble_result(cfgs[4], self.exp_lst5, test_epoch=300)
        inds_df6 = load_ensemble_result(cfgs[5], self.exp_lst6, test_epoch=300)
        xs2 = []
        ys2 = []
        cases_exps_legends_together_2 = ["3557_cet", "3557_et", "3557_no-et"]

        x4, y4 = ecdf(np.nan_to_num(inds_df4[keys_nse]))
        xs2.append(x4)
        ys2.append(y4)

        x5, y5 = ecdf(np.nan_to_num(inds_df5[keys_nse]))
        xs2.append(x5)
        ys2.append(y5)

        x6, y6 = ecdf(np.nan_to_num(inds_df6[keys_nse]))
        xs2.append(x6)
        ys2.append(y6)

        plot_ecdfs_matplot(xs2, ys2, cases_exps_legends_together_2,
                           colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
                           dash_lines=[False, False, False], x_str="NSE", y_str="CDF")
        plt.savefig(os.path.join(config_datas[0].data_path["Out"], '3557_cet_et_no-et.png'), dpi=600,
                    bbox_inches="tight")

        xs3 = []
        ys3 = []
        cases_exps_legends_together_3 = ["328in3557_cet", "328in3557_et", "328in3557_no-et"]
        idx_lst_328in3557 = [i for i in range(len(self.basins_id)) if self.basins_id[i] in self.irri_basins_id]
        x7, y7 = ecdf(np.nan_to_num(inds_df4[keys_nse].iloc[idx_lst_328in3557]))
        xs3.append(x7)
        ys3.append(y7)

        x8, y8 = ecdf(np.nan_to_num(inds_df5[keys_nse].iloc[idx_lst_328in3557]))
        xs3.append(x8)
        ys3.append(y8)

        x9, y9 = ecdf(np.nan_to_num(inds_df6[keys_nse].iloc[idx_lst_328in3557]))
        xs3.append(x9)
        ys3.append(y9)

        plot_ecdfs_matplot(xs3, ys3, cases_exps_legends_together_3,
                           colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
                           dash_lines=[False, False, False], x_str="NSE", y_str="CDF")
        plt.savefig(os.path.join(config_datas[0].data_path["Out"], '328in3557_cet_et_no-et.png'), dpi=600,
                    bbox_inches="tight")


if __name__ == '__main__':
    unittest.main()
