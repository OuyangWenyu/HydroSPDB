import unittest
import numpy as np
import pandas as pd
import definitions
from data import *
import os
from data.data_input import GagesModel, load_result
from explore.gages_stat import split_results_to_regions
from explore.stat import statError
from utils import unserialize_json
from utils.dataset_format import subset_of_dict
from visual.plot_model import plot_gages_map_and_ts, plot_gages_attrs_boxes, plot_scatter_multi_attrs
from visual.plot_stat import plot_diff_boxes, plot_scatter_xyc, plot_boxs, swarmplot_with_cbar
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns


class TestExploreCase(unittest.TestCase):
    def setUp(self):
        """analyze result of model"""
        config_dir = definitions.CONFIG_DIR
        # self.ref_config_file = os.path.join(config_dir, "basic/config_exp2.ini")
        # self.ref_subdir = r"basic/exp2"
        # self.ref_config_data = GagesConfig.set_subdir(self.ref_config_file, self.ref_subdir)
        # self.nonref_config_file = os.path.join(config_dir, "basic/config_exp18.ini")
        # self.nonref_subdir = r"basic/exp18"
        # self.nonref_config_data = GagesConfig.set_subdir(self.nonref_config_file, self.nonref_subdir)
        self.nomajordam_config_file = os.path.join(config_dir, "nodam/config_exp2.ini")
        self.nomajordam_subdir = r"nodam/exp2"
        self.nomajordam_config_data = GagesConfig.set_subdir(self.nomajordam_config_file, self.nomajordam_subdir)
        self.majordam_config_file = os.path.join(config_dir, "majordam/config_exp1.ini")
        self.majordam_subdir = r"majordam/exp1"
        self.majordam_config_data = GagesConfig.set_subdir(self.majordam_config_file, self.majordam_subdir)
        self.config_file = os.path.join(config_dir, "basic/config_exp23.ini")
        self.subdir = r"basic/exp23"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.test_epoch = 300

        self.data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')

        self.nomajordam_data_model = GagesModel.load_datamodel(self.nomajordam_config_data.data_path["Temp"],
                                                               data_source_file_name='test_data_source.txt',
                                                               stat_file_name='test_Statistics.json',
                                                               flow_file_name='test_flow.npy',
                                                               forcing_file_name='test_forcing.npy',
                                                               attr_file_name='test_attr.npy',
                                                               f_dict_file_name='test_dictFactorize.json',
                                                               var_dict_file_name='test_dictAttribute.json',
                                                               t_s_dict_file_name='test_dictTimeSpace.json')

        self.majordam_data_model = GagesModel.load_datamodel(self.majordam_config_data.data_path["Temp"],
                                                             data_source_file_name='test_data_source.txt',
                                                             stat_file_name='test_Statistics.json',
                                                             flow_file_name='test_flow.npy',
                                                             forcing_file_name='test_forcing.npy',
                                                             attr_file_name='test_attr.npy',
                                                             f_dict_file_name='test_dictFactorize.json',
                                                             var_dict_file_name='test_dictAttribute.json',
                                                             t_s_dict_file_name='test_dictTimeSpace.json')

        attrBasin = ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM']
        attrLandcover = ['FORESTNLCD06', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06',
                         'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
        attrSoil = ['ROCKDEPAVE', 'AWCAVE', 'PERMAVE', 'RFACT']
        attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_SITE']
        attrHydro = ['STREAMS_KM_SQ_KM', 'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
                     'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'BFI_AVE', 'PERDUN', 'PERHOR', 'TOPWET', 'CONTACT']
        attrHydroModDams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_AVG_DIS_ALL_MAJ_DAMS']
        attrHydroModOther = ['CANALS_PCT', 'RAW_AVG_DIS_ALLCANALS',
                             'NPDES_MAJ_DENS', 'RAW_AVG_DIS_ALL_MAJ_NPDES', 'FRESHW_WITHDRAWAL',
                             'PCT_IRRIG_AG', 'POWER_SUM_MW']
        attrLandscapePat = ['FRAGUN_BASIN']
        attrLC06Basin = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06']
        attrPopInfrastr = ['ROADS_KM_SQ_KM']
        attrProtAreas = ['PADCAT1_PCT_BASIN', 'PADCAT2_PCT_BASIN']
        self.attr_lst = attrLandscapePat + attrLC06Basin + attrPopInfrastr + attrProtAreas
        # self.attr_lst = attrHydroModOther

        # plot is_nse_good
        pred, obs = load_result(self.data_model.data_source.data_config.data_path['Temp'], self.test_epoch)
        self.pred = pred.reshape(pred.shape[0], pred.shape[1])
        self.obs = obs.reshape(pred.shape[0], pred.shape[1])
        inds = statError(self.obs, self.pred)
        self.inds_df = pd.DataFrame(inds)

    def test_analyze_isref(self):
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        sites_ref = self.nomajordam_data_model.t_s_dict["sites_id"]
        attrs_nonref = self.data_model.data_source.read_attr(sites_nonref, self.attr_lst, is_return_dict=False)
        attrs_ref = self.data_model.data_source.read_attr(sites_ref, self.attr_lst, is_return_dict=False)
        plot_gages_attrs_boxes(sites_nonref, sites_ref, self.attr_lst, attrs_nonref, attrs_ref, diff_str="IS_REF",
                               row_and_col=[2, 4])

    def test_analyze_isnsegood(self):
        inds_df = self.inds_df
        nse_below = 0.5
        show_ind_key = 'NSE'
        idx_lst_small_nse = inds_df[(inds_df[show_ind_key] < nse_below)].index.tolist()
        sites_small_nse = np.array(self.data_model.t_s_dict['sites_id'])[idx_lst_small_nse]
        assert (all(x < y for x, y in zip(sites_small_nse, sites_small_nse[1:])))
        idx_lst_big_nse = inds_df[(inds_df[show_ind_key] >= nse_below)].index.tolist()
        sites_big_nse = np.array(self.data_model.t_s_dict['sites_id'])[idx_lst_big_nse]
        assert (all(x < y for x, y in zip(sites_big_nse, sites_big_nse[1:])))

        attrs_small = self.data_model.data_source.read_attr(sites_small_nse, self.attr_lst, is_return_dict=False)
        attrs_big = self.data_model.data_source.read_attr(sites_big_nse, self.attr_lst, is_return_dict=False)
        plot_gages_attrs_boxes(sites_small_nse, sites_big_nse, self.attr_lst, attrs_small, attrs_big,
                               diff_str="IS_NSE_GOOD", row_and_col=[2, 4])

    def test_map_ts(self):
        # plot map ts
        inds_df = self.inds_df
        show_ind_key = 'NSE'
        idx_lst = np.arange(len(self.data_model.t_s_dict["sites_id"])).tolist()

        nse_range = [0.5, 1]
        idx_lst_small_nse = inds_df[
            (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()
        plot_gages_map_and_ts(self.data_model, self.obs, self.pred, inds_df, show_ind_key, idx_lst_small_nse,
                              pertile_range=[0, 100])

    def test_x_y_scatter(self):
        elev_var = "ELEV_MEAN_M_BASIN"
        attr_elev_lst = [elev_var]
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        attrs_elev = self.data_model.data_source.read_attr(sites_nonref, attr_elev_lst, is_return_dict=False)
        inds_df_now = self.inds_df
        nse_range = [0, 1]
        show_ind_key = 'NSE'
        idx_lst_nse_range = inds_df_now[
            (inds_df_now[show_ind_key] >= nse_range[0]) & (inds_df_now[show_ind_key] < nse_range[1])].index.tolist()
        nse_values = self.inds_df["NSE"].values[idx_lst_nse_range]
        df = pd.DataFrame({elev_var: attrs_elev[idx_lst_nse_range, 0], show_ind_key: nse_values})
        g = sns.jointplot(x=elev_var, y=show_ind_key, data=df, kind="reg")
        g.ax_marg_x.set_xlim(0, 3000)
        g.ax_marg_y.set_ylim(0, 1)
        plt.show()

    def test_x_y_color_scatter(self):
        elev_var = "ELEV_MEAN_M_BASIN"
        stor_var = "STOR_NOR_2009"
        attr_elev_stor_lst = [elev_var, stor_var]
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        attrs_elev_stor = self.data_model.data_source.read_attr(sites_nonref, attr_elev_stor_lst, is_return_dict=False)

        inds_df_now = self.inds_df
        nse_range = [0, 1]
        show_ind_key = 'NSE'
        idx_lst_nse_range = inds_df_now[
            (inds_df_now[show_ind_key] >= nse_range[0]) & (inds_df_now[show_ind_key] < nse_range[1])].index.tolist()
        nse_values = self.inds_df["NSE"].values[idx_lst_nse_range]

        plot_scatter_xyc(elev_var, attrs_elev_stor[idx_lst_nse_range, 0], stor_var,
                         attrs_elev_stor[idx_lst_nse_range, 1], c_label=show_ind_key, c=nse_values, is_reg=True,
                         ylim=[0, 2000])
        # plt.xlabel(elev_var + '(m)')
        # plt.ylabel(stor_var + '(1000 m^3/sq km)')

    def test_scatters(self):
        attr_lst = self.attr_lst
        show_ind_key = 'NSE'
        y_var_lst = [show_ind_key]
        inds_df_now = self.inds_df
        nse_range = [0, 1]
        # idx_lst_nse_range = inds_df_now.index.tolist()
        idx_lst_nse_range = inds_df_now[
            (inds_df_now[show_ind_key] >= nse_range[0]) & (inds_df_now[show_ind_key] < nse_range[1])].index.tolist()
        plot_scatter_multi_attrs(self.data_model, self.inds_df, idx_lst_nse_range, attr_lst, y_var_lst)

    def test_scatter_dor(self):
        attr_lst = ["RUNAVE7100", "STOR_NOR_2009"]
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        attrs_runavg_stor = self.data_model.data_source.read_attr(sites_nonref, attr_lst, is_return_dict=False)
        run_avg = attrs_runavg_stor[:, 0] * (10 ** (-3)) * (10 ** 6)  # m^3 per year
        nor_storage = attrs_runavg_stor[:, 1] * 1000  # m^3
        dors = nor_storage / run_avg
        inds_df_now = self.inds_df
        nse_range = [0, 1]
        show_ind_key = 'NSE'
        idx_lst_nse_range = inds_df_now[
            (inds_df_now[show_ind_key] >= nse_range[0]) & (inds_df_now[show_ind_key] < nse_range[1])].index.tolist()
        nse_values = self.inds_df["NSE"].values[idx_lst_nse_range]
        df = pd.DataFrame({"DOR": dors[idx_lst_nse_range], show_ind_key: nse_values})
        plot_scatter_xyc("DOR", dors[idx_lst_nse_range], show_ind_key, nse_values, is_reg=True, ylim=[0, 1])
        # g = sns.jointplot(x="DOR", y=show_ind_key, data=df, kind="reg")
        # g.ax_marg_x.set_xlim(0, 1)
        # g.ax_marg_y.set_ylim(0, 1)
        plt.show()

    def test_scatter_dam_purpose(self):
        attr_lst = ["RUNAVE7100", "STOR_NOR_2009"]
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        attrs_runavg_stor = self.data_model.data_source.read_attr(sites_nonref, attr_lst, is_return_dict=False)
        run_avg = attrs_runavg_stor[:, 0] * (10 ** (-3)) * (10 ** 6)  # m^3 per year
        nor_storage = attrs_runavg_stor[:, 1] * 1000  # m^3
        dors = nor_storage / run_avg

        nid_dir = os.path.join("/".join(self.config_data.data_path["DB"].split("/")[:-1]), "nid", "quickdata")
        gage_main_dam_purpose = unserialize_json(os.path.join(nid_dir, "dam_main_purpose_dict.json"))
        gage_main_dam_purpose_lst = list(gage_main_dam_purpose.values())
        gage_main_dam_purpose_unique = np.unique(gage_main_dam_purpose_lst)
        purpose_regions = {}
        for i in range(gage_main_dam_purpose_unique.size):
            sites_id = []
            for key, value in gage_main_dam_purpose.items():
                if value == gage_main_dam_purpose_unique[i]:
                    sites_id.append(key)
            assert (all(x < y for x, y in zip(sites_id, sites_id[1:])))
            purpose_regions[gage_main_dam_purpose_unique[i]] = sites_id
        id_regions_idx = []
        id_regions_sites_ids = []
        regions_name = []
        show_min_num = 10
        df_id_region = np.array(self.data_model.t_s_dict["sites_id"])
        for key, value in purpose_regions.items():
            gages_id = value
            c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
            if c.size < show_min_num:
                continue
            assert (all(x < y for x, y in zip(ind1, ind1[1:])))
            assert (all(x < y for x, y in zip(c, c[1:])))
            id_regions_idx.append(ind1)
            id_regions_sites_ids.append(c)
            regions_name.append(key)
        preds, obss, inds_dfs = split_results_to_regions(self.data_model, self.test_epoch, id_regions_idx,
                                                         id_regions_sites_ids)
        frames = []
        x_name = "purposes"
        y_name = "NSE"
        hue_name = "DOR"
        # hue_name = "STOR"
        for i in range(len(id_regions_idx)):
            # plot box，使用seaborn库
            keys = ["NSE"]
            inds_test = subset_of_dict(inds_dfs[i], keys)
            inds_test = inds_test[keys[0]].values
            df_dict_i = {}
            str_i = regions_name[i]
            df_dict_i[x_name] = np.full([inds_test.size], str_i)
            df_dict_i[y_name] = inds_test
            df_dict_i[hue_name] = dors[id_regions_idx[i]]
            # df_dict_i[hue_name] = nor_storage[id_regions_idx[i]]
            df_i = pd.DataFrame(df_dict_i)
            frames.append(df_i)
        result = pd.concat(frames)
        # can remove high hue value to keep a good map
        # plot_boxs(result, x_name, y_name, uniform_color="skyblue", swarm_plot=True, hue=hue_name, colormap=True,
        #           ylim=[-0.6, 1.0])
        cmap_str = 'viridis'
        # cmap = plt.get_cmap('Spectral')
        cbar_label = hue_name

        plt.title('Distribution of different purposes')
        swarmplot_with_cbar(cmap_str, cbar_label, [-1, 1.0], x=x_name, y=y_name, hue=hue_name, palette=cmap_str,
                            data=result)
        # swarmplot_with_cbar(cmap_str, cbar_label, None, x=x_name, y=y_name, hue=hue_name, palette=cmap_str, data=result)

    def test_explore_gages_prcp_log(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='data_source.txt',
                                               stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                               forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                               f_dict_file_name='dictFactorize.json',
                                               var_dict_file_name='dictAttribute.json',
                                               t_s_dict_file_name='dictTimeSpace.json')
        i = np.random.randint(data_model.data_forcing.shape[0], size=1)
        print(i)
        a = data_model.data_forcing[i, :, 1].flatten()
        series = a[~np.isnan(a)]
        series = series[np.where(series >= 0)]
        # series = series[np.where(series > 0)]
        pyplot.plot(series)
        pyplot.show()
        # histogram
        pyplot.hist(series)
        pyplot.show()
        # sqrt transform
        transform = np.sqrt(series)
        pyplot.figure(1)
        # line plot
        pyplot.subplot(211)
        pyplot.plot(transform)
        # histogram
        pyplot.subplot(212)
        pyplot.hist(transform)
        pyplot.show()
        transform = np.log(series + 1)
        # transform = np.log(series + 0.1)
        pyplot.figure(1)
        # line plot
        pyplot.subplot(211)
        pyplot.plot(transform)
        # histogram
        pyplot.subplot(212)
        pyplot.hist(transform)
        pyplot.show()
        # transform = stats.boxcox(series, lmbda=0.0)
        # pyplot.figure(1)
        # # line plot
        # pyplot.subplot(211)
        # pyplot.plot(transform)
        # # histogram
        # pyplot.subplot(212)
        # pyplot.hist(transform)
        # pyplot.show()
        # for j in range(data_model.data_forcing.shape[2]):
        #     x_explore_j = data_model.data_forcing[i, :, j].flatten()
        #     plot_dist(x_explore_j)

    def test_explore_gages_prcp_basin(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='data_source.txt',
                                               stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                               forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                               f_dict_file_name='dictFactorize.json',
                                               var_dict_file_name='dictAttribute.json',
                                               t_s_dict_file_name='dictTimeSpace.json')
        basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                      is_return_dict=False)
        mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                     is_return_dict=False)
        flow = data_model.data_flow
        temparea = np.tile(basin_area, (1, flow.shape[1]))
        tempprep = np.tile(mean_prep / 365 * 10, (1, flow.shape[1]))
        flowua = (flow * 0.0283168 * 3600 * 24) / (
                (temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))  # unit (m^3/day)/(m^3/day)
        i = np.random.randint(data_model.data_forcing.shape[0], size=1)
        a = flow[i].flatten()
        series = a[~np.isnan(a)]
        # series = series[np.where(series >= 0)]
        # series = series[np.where(series > 0)]
        pyplot.figure(1)
        # line plot
        pyplot.subplot(211)
        pyplot.plot(series)
        # histogram
        pyplot.subplot(212)
        pyplot.hist(series)
        pyplot.show()

        b = flowua[i].flatten()
        transform = b[~np.isnan(b)]
        # transform = series[np.where(transform >= 0)]
        # series = series[np.where(series > 0)]
        pyplot.figure(1)
        # line plot
        pyplot.subplot(211)
        pyplot.plot(transform)
        # histogram
        pyplot.subplot(212)
        pyplot.hist(transform)
        pyplot.show()


if __name__ == '__main__':
    unittest.main()
