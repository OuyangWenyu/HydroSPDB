"""for stacked lstm"""
import copy
import operator
import os
import pandas as pd
import torch
import numpy as np
import geopandas as gpd

from data import DataModel, GagesSource
from data.data_config import update_config_item
from data.data_input import GagesModel
from explore import trans_norm, cal_stat
from explore.hydro_cluster import cluster_attr_train
from hydroDL import master_train
from hydroDL.model import model_run
from utils.dataset_format import subset_of_dict
from utils.hydro_math import concat_two_3darray, copy_attr_array_in2d


class GagesModels(object):
    """the data model for GAGES-II dataset"""

    def __init__(self, config_data):
        # 准备训练数据
        t_train = config_data.model_dict["data"]["tRangeTrain"]
        t_test = config_data.model_dict["data"]["tRangeTest"]
        t_train_test = [t_train[0], t_test[1]]
        source_data = GagesSource(config_data, t_train_test)
        # 构建输入数据类对象
        data_model = GagesModel(source_data)
        self.data_model_train, self.data_model_test = GagesModel.data_models_of_train_test(data_model, t_train, t_test)


class GagesInvDataModel(object):
    """DataModel for inv model"""

    def __init__(self, data_model1, data_model2):
        self.model_dict1 = data_model1.data_source.data_config.model_dict
        self.model_dict2 = data_model2.data_source.data_config.model_dict
        self.stat_dict = data_model2.stat_dict
        self.t_s_dict = data_model2.t_s_dict
        all_data = self.prepare_input(data_model1, data_model2)
        input_keys = ['xh', 'ch', 'qh', 'xt', 'ct']
        output_keys = ['qt']
        self.data_input = subset_of_dict(all_data, input_keys)
        self.data_target = subset_of_dict(all_data, output_keys)

    def prepare_input(self, data_model1, data_model2):
        """prepare input for lstm-inv, gages_id may be different, fix it here"""
        print("prepare input")
        sites_id1 = data_model1.t_s_dict['sites_id']
        sites_id2 = data_model2.t_s_dict['sites_id']
        sites_id, ind1, ind2 = np.intersect1d(sites_id1, sites_id2, return_indices=True)
        data_model1.data_attr = data_model1.data_attr[ind1, :]
        data_model1.data_flow = data_model1.data_flow[ind1, :]
        data_model1.data_forcing = data_model1.data_forcing[ind1, :]
        data_model2.data_attr = data_model2.data_attr[ind2, :]
        data_model2.data_flow = data_model2.data_flow[ind2, :]
        data_model2.data_forcing = data_model2.data_forcing[ind2, :]
        data_model1.t_s_dict['sites_id'] = sites_id
        data_model2.t_s_dict['sites_id'] = sites_id
        model_dict1 = data_model1.data_source.data_config.model_dict
        xh, qh, ch = data_model1.load_data(model_dict1)
        model_dict2 = data_model2.data_source.data_config.model_dict
        xt, qt, ct = data_model2.load_data(model_dict2)
        return {'xh': xh, 'ch': ch, 'qh': qh, 'xt': xt, 'ct': ct, 'qt': qt}

    def load_data(self):
        data_input = self.data_input
        data_inflow_h = data_input['qh']
        data_inflow_h = data_inflow_h.reshape(data_inflow_h.shape[0], data_inflow_h.shape[1])
        # transform x to 3d, the final dim's length is the seq_length
        seq_length = self.model_dict1["model"]["seqLength"]
        data_inflow_h_new = np.zeros([data_inflow_h.shape[0], data_inflow_h.shape[1] - seq_length + 1, seq_length])
        for i in range(data_inflow_h_new.shape[1]):
            data_inflow_h_new[:, i, :] = data_inflow_h[:, i:i + seq_length]

        # because data_inflow_h_new is assimilated, time sequence length has changed
        data_forcing_h = data_input['xh'][:, seq_length - 1:, :]
        xqh = concat_two_3darray(data_inflow_h_new, data_forcing_h)

        attr_h = data_input['ch']
        attr_h_new = copy_attr_array_in2d(attr_h, xqh.shape[1])

        # concatenate xqh with ch
        xqch = concat_two_3darray(xqh, attr_h_new)

        # concatenate xt with ct
        data_forcing_t = data_input['xt']
        attr_t = data_input['ct']
        attr_t_new = copy_attr_array_in2d(attr_t, data_forcing_t.shape[1])
        xct = concat_two_3darray(data_forcing_t, attr_t_new)

        qt = self.data_target["qt"]
        return xqch, xct, qt


class GagesSimInvDataModel(object):
    """DataModel for siminv model"""

    def __init__(self, data_model1, data_model2, data_model3):
        all_data = self.prepare_input(data_model1, data_model2, data_model3)
        input_keys = ['xh', 'ch', 'qh', 'qnh', 'xt', 'ct']
        output_keys = ['qt']
        self.data_input = subset_of_dict(all_data, input_keys)
        self.data_target = subset_of_dict(all_data, output_keys)
        self.model_dict2 = data_model2.data_source.data_config.model_dict
        self.test_trange = data_model3.t_s_dict["t_final_range"]
        self.test_stat_dict = data_model3.stat_dict
        self.test_t_s_dict = data_model3.t_s_dict

    def read_natural_inflow(self, sim_model_data, model_data):
        sim_config_data = sim_model_data.data_source.data_config
        # read model
        # firstly, check if the model used to generate natural flow has existed
        out_folder = sim_config_data.data_path["Out"]
        epoch = sim_config_data.model_dict["train"]["nEpoch"]
        model_file = os.path.join(out_folder, 'model_Ep' + str(epoch) + '.pt')
        if not os.path.isfile(model_file):
            master_train(sim_model_data)
        model = torch.load(model_file)
        # run the model
        config_data = model_data.data_source.data_config
        model_dict = config_data.model_dict
        batch_size = model_dict["train"]["miniBatch"][0]
        x, y, c = model_data.load_data(model_dict)
        t_range = model_data.t_s_dict["t_final_range"]
        natural_epoch = model_dict["train"]["nEpoch"]
        file_name = '_'.join([str(t_range[0]), str(t_range[1]), 'ep' + str(natural_epoch)])
        file_path = os.path.join(out_folder, file_name) + '.csv'
        model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
        # read natural_flow from file
        np_natural_flow = pd.read_csv(file_path, dtype=np.float, header=None).values
        return np_natural_flow

    def prepare_input(self, data_model1, data_model2, data_model3):
        """prepare input for lstm-inv, gages_id may be different, fix it here"""
        print("prepare input")
        sim_flow = self.read_natural_inflow(data_model1, data_model2)
        sites_id2 = data_model2.t_s_dict['sites_id']
        sites_id3 = data_model3.t_s_dict['sites_id']
        sites_id, ind1, ind2 = np.intersect1d(sites_id2, sites_id3, return_indices=True)
        data_model2.data_attr = data_model2.data_attr[ind1, :]
        data_model2.data_flow = data_model2.data_flow[ind1, :]
        data_model2.data_forcing = data_model2.data_forcing[ind1, :]
        data_model3.data_attr = data_model3.data_attr[ind2, :]
        data_model3.data_flow = data_model3.data_flow[ind2, :]
        data_model3.data_forcing = data_model3.data_forcing[ind2, :]
        data_model2.t_s_dict['sites_id'] = sites_id
        data_model3.t_s_dict['sites_id'] = sites_id
        qnh = np.expand_dims(sim_flow[ind1, :], axis=2)
        model_dict2 = data_model2.data_source.data_config.model_dict
        xh, qh, ch = data_model2.load_data(model_dict2)
        model_dict3 = data_model3.data_source.data_config.model_dict
        xt, qt, ct = data_model3.load_data(model_dict3)
        return {'xh': xh, 'ch': ch, 'qh': qh, 'qnh': qnh, 'xt': xt, 'ct': ct, 'qt': qt}

    def load_data(self):
        data_input = self.data_input
        data_inflow_h = data_input['qh']
        data_nat_inflow_h = data_input['qnh']
        seq_length = self.model_dict2["model"]["seqLength"]

        def trans_to_tim_seq(data_now, seq_length_now):
            data_now = data_now.reshape(data_now.shape[0], data_now.shape[1])
            # the final dim's length is the seq_length
            data_now_new = np.zeros([data_now.shape[0], data_now.shape[1] - seq_length_now + 1, seq_length_now])
            for i in range(data_now_new.shape[1]):
                data_now_new[:, i, :] = data_now[:, i:i + seq_length_now]
            return data_now_new

        data_inflow_h_new = trans_to_tim_seq(data_inflow_h, seq_length)
        data_nat_inflow_h_new = trans_to_tim_seq(data_nat_inflow_h, seq_length)
        qqnh = concat_two_3darray(data_inflow_h_new, data_nat_inflow_h_new)
        # because data_inflow_h_new is assimilated, time sequence length has changed
        data_forcing_h = data_input['xh'][:, seq_length - 1:, :]
        xqqnh = concat_two_3darray(qqnh, data_forcing_h)

        def copy_attr_array_in2d(arr1, len_of_2d):
            arr2 = np.zeros([arr1.shape[0], len_of_2d, arr1.shape[1]])
            for k in range(arr1.shape[0]):
                arr2[k] = np.tile(arr1[k], arr2.shape[1]).reshape(arr2.shape[1], arr1.shape[1])
            return arr2

        attr_h = data_input['ch']
        attr_h_new = copy_attr_array_in2d(attr_h, xqqnh.shape[1])

        # concatenate xqh with ch
        xqqnch = concat_two_3darray(xqqnh, attr_h_new)

        # concatenate xt with ct
        data_forcing_t = data_input['xt']
        attr_t = data_input['ct']
        attr_t_new = copy_attr_array_in2d(attr_t, data_forcing_t.shape[1])
        xct = concat_two_3darray(data_forcing_t, attr_t_new)

        qt = self.data_target["qt"]
        return xqqnch, xct, qt


class GagesDaDataModel(object):
    """DataModel for da model"""

    def __init__(self, data_model):
        self.data_model = data_model

    def load_data(self, model_dict):
        """Notice that don't cover the data of today when loading history data"""

        def cut_data(temp_x, temp_rm_nan, temp_seq_length):
            """cut to size same as inflow's. Don't cover the data of today when loading history data"""
            temp = temp_x[:, temp_seq_length:, :]
            if temp_rm_nan:
                temp[np.where(np.isnan(temp))] = 0
            return temp

        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        x, y, c = self.data_model.load_data(model_dict)

        seq_length = model_dict["model"]["seqLength"]
        # don't cover the data of today when loading history data, so the length of 2nd dim is 'y.shape[1] - seq_length'
        flow = y.reshape(y.shape[0], y.shape[1])
        q = np.zeros([flow.shape[0], flow.shape[1] - seq_length, seq_length])
        for i in range(q.shape[1]):
            q[:, i, :] = flow[:, i:i + seq_length]

        if rm_nan_x is True:
            q[np.where(np.isnan(q))] = 0

        if seq_length > 1:
            x = cut_data(x, rm_nan_x, seq_length)
            y = cut_data(y, rm_nan_y, seq_length)
        qx = np.array([np.concatenate((q[j], x[j]), axis=1) for j in range(q.shape[0])])
        return qx, y, c


class GagesForecastDataModel(object):
    """DataModel for assimilation of forecast data"""

    def __init__(self, sim_data_model, data_model):
        self.sim_data_model = sim_data_model
        self.model_data = data_model
        self.natural_flow = self.read_natural_inflow_and_forecast()

    def read_natural_inflow_and_forecast(self):
        sim_model_data = self.sim_data_model
        sim_config_data = sim_model_data.data_source.data_config
        # read model
        # firstly, check if the model used to generate natural flow has existed
        out_folder = sim_config_data.data_path["Out"]
        epoch = sim_config_data.model_dict["train"]["nEpoch"]
        model_file = os.path.join(out_folder, 'model_Ep' + str(epoch) + '.pt')
        if not os.path.isfile(model_file):
            master_train(sim_model_data)
        model = torch.load(model_file)
        # run the model
        model_data = self.model_data
        config_data = model_data.data_source.data_config
        model_dict = config_data.model_dict
        batch_size = model_dict["train"]["miniBatch"][0]
        x, y, c = model_data.load_data(model_dict)
        t_range = self.model_data.t_s_dict["t_final_range"]
        natural_epoch = model_dict["train"]["nEpoch"]
        file_name = '_'.join([str(t_range[0]), str(t_range[1]), 'ep' + str(natural_epoch)])
        file_path = os.path.join(out_folder, file_name) + '.csv'
        model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
        # read natural_flow from file
        np_natural_flow = pd.read_csv(file_path, dtype=np.float, header=None).values
        return np_natural_flow

    def load_data(self, model_dict):
        """Notice that don't cover the data of today when loading history data"""

        def cut_data(temp_x, temp_rm_nan, temp_seq_length, temp_fcst_length):
            """cut to size same as inflow's. Cover future natural flow"""
            temp = temp_x[:, temp_seq_length - 1: -temp_fcst_length, :]
            if temp_rm_nan:
                temp[np.where(np.isnan(temp))] = 0
            return temp

        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        x, y, c = self.model_data.load_data(model_dict)

        seq_length = model_dict["model"]["seqLength"]
        fcst_length = model_dict["model"]["fcstLength"]
        # don't cover the data of today when loading history data, so the length of 2nd dim is 'y.shape[1] - seq_length'
        flow = self.natural_flow
        q = np.zeros([flow.shape[0], flow.shape[1] - seq_length - fcst_length + 1, seq_length + fcst_length])
        for i in range(q.shape[1]):
            q[:, i, :] = flow[:, i:i + seq_length + fcst_length]

        if rm_nan_x is True:
            q[np.where(np.isnan(q))] = 0

        if seq_length >= 1 or fcst_length >= 1:
            x = cut_data(x, rm_nan_x, seq_length, fcst_length)
            y = cut_data(y, rm_nan_y, seq_length, fcst_length)
        qx = np.array([np.concatenate((q[j], x[j]), axis=1) for j in range(q.shape[0])])
        return qx, y, c


def divide_to_classes(label_dict, model_data, num_cluster, sites_id_all, with_dam_purpose=False):
    data_models = []
    var_dict = model_data.var_dict
    f_dict = model_data.f_dict
    for i in range(num_cluster):
        sites_label_i = [key for key, value in label_dict.items() if value == i]
        sites_label_i_index = [j for j in range(len(sites_id_all)) if sites_id_all[j] in sites_label_i]
        data_flow = model_data.data_flow[sites_label_i_index, :]
        data_forcing = model_data.data_forcing[sites_label_i_index, :, :]
        data_attr = model_data.data_attr[sites_label_i_index, :]
        stat_dict = {}
        t_s_dict = {}
        source_data_i = copy.deepcopy(model_data.data_source)
        out_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Out"], str(i))
        if not os.path.isdir(out_dir_new):
            os.makedirs(out_dir_new)
        temp_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Temp"], str(i))
        if not os.path.isdir(temp_dir_new):
            os.makedirs(temp_dir_new)
        update_config_item(source_data_i.data_config.data_path, Out=out_dir_new, Temp=temp_dir_new)
        update_config_item(source_data_i.all_configs, out_dir=out_dir_new, temp_dir=temp_dir_new,
                           flow_screen_gage_id=sites_label_i)
        f_dict_new = copy.deepcopy(f_dict)
        if with_dam_purpose:
            if num_cluster > len(f_dict['GAGE_MAIN_DAM_PURPOSE']):
                # there is a "None" type
                if i == 0:
                    f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = None
                else:
                    f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = [f_dict['GAGE_MAIN_DAM_PURPOSE'][i - 1]]
            else:
                f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = [f_dict['GAGE_MAIN_DAM_PURPOSE'][i]]
        data_model_i = DataModel(source_data_i, data_flow, data_forcing, data_attr, var_dict, f_dict_new, stat_dict,
                                 t_s_dict)
        t_s_dict['sites_id'] = sites_label_i
        t_s_dict['t_final_range'] = source_data_i.t_range
        data_model_i.t_s_dict = t_s_dict
        stat_dict_i = data_model_i.cal_stat_all()
        data_model_i.stat_dict = stat_dict_i
        data_models.append(data_model_i)
    return data_models


class GagesExploreDataModel(object):
    def __init__(self, data_model):
        self.data_model = data_model

    def cluster_datamodel(self, num_cluster, start_dam_var='NDAMS_2009', with_dam_purpose=False,
                          sites_ids_list=None):
        """according to attr, cluster dataset"""
        model_data = self.data_model
        sites_id_all = model_data.t_s_dict["sites_id"]
        label_dict = {}
        if sites_ids_list:
            for k in range(len(sites_ids_list)):
                for site_id_temp in sites_ids_list[k]:
                    label_dict[site_id_temp] = k
        else:
            stat_dict = model_data.stat_dict
            var_lst = model_data.data_source.all_configs.get("attr_chosen")
            data = trans_norm(model_data.data_attr, var_lst, stat_dict, to_norm=True)
            index_start_anthro = 0
            for i in range(len(var_lst)):
                if var_lst[i] == start_dam_var:
                    index_start_anthro = i
                    break
            norm_data = data[:, index_start_anthro:]
            kmeans, labels = cluster_attr_train(norm_data, num_cluster)
            label_dict = dict(zip(sites_id_all, labels))
        if with_dam_purpose:
            data_models = divide_to_classes(label_dict, model_data, num_cluster, sites_id_all,
                                            with_dam_purpose=True)
        else:
            data_models = divide_to_classes(label_dict, model_data, num_cluster, sites_id_all)
        return data_models

    def classify_datamodel_by_dam_purpose(self, sites_ids_list=None):
        """classify data into classes one of which include all gage with same main dam purpose"""
        model_data = self.data_model
        sites_id_all = model_data.t_s_dict["sites_id"]
        data_attrs = model_data.data_attr[:, -1]
        data_attrs_unique, indices = np.unique(data_attrs, return_index=True)
        label_dict = {}
        if sites_ids_list:
            for k in range(len(sites_ids_list)):
                for site_id_temp in sites_ids_list[k]:
                    label_dict[site_id_temp] = k
        else:
            for i in range(data_attrs_unique.size):
                for j in range(len(sites_id_all)):
                    if data_attrs[j] == data_attrs_unique[i]:
                        label_dict[sites_id_all[j]] = i
        data_models = divide_to_classes(label_dict, model_data, data_attrs_unique.size, sites_id_all,
                                        with_dam_purpose=True)
        return data_models

    def choose_datamodel(self, sites_ids, f_dict_dam_purpose, sub_dir_num):
        model_data = self.data_model
        sites_id_all = model_data.t_s_dict["sites_id"]
        sites_label_i_index = [j for j in range(len(sites_id_all)) if sites_id_all[j] in sites_ids]
        data_flow = model_data.data_flow[sites_label_i_index, :]
        data_forcing = model_data.data_forcing[sites_label_i_index, :, :]
        data_attr = model_data.data_attr[sites_label_i_index, :]
        stat_dict = {}
        t_s_dict = {}
        source_data_i = copy.deepcopy(model_data.data_source)
        out_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Out"], str(sub_dir_num))
        temp_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Temp"], str(sub_dir_num))
        update_config_item(source_data_i.data_config.data_path, Out=out_dir_new, Temp=temp_dir_new)
        sites_id_all_np = np.array(sites_id_all)
        update_config_item(source_data_i.all_configs, out_dir=out_dir_new, temp_dir=temp_dir_new,
                           flow_screen_gage_id=sites_id_all_np[sites_label_i_index].tolist())
        f_dict = model_data.f_dict
        f_dict_new = copy.deepcopy(f_dict)
        if f_dict_dam_purpose is None:
            f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = None
        else:
            f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = f_dict_dam_purpose
        var_dict = model_data.var_dict
        data_model_i = DataModel(source_data_i, data_flow, data_forcing, data_attr, var_dict, f_dict_new, stat_dict,
                                 t_s_dict)
        t_s_dict['sites_id'] = sites_id_all_np[sites_label_i_index].tolist()
        t_s_dict['t_final_range'] = source_data_i.t_range
        data_model_i.t_s_dict = t_s_dict
        stat_dict_i = data_model_i.cal_stat_all()
        data_model_i.stat_dict = stat_dict_i
        return data_model_i

    def choose_datamodel_nodam(self, sites_ids, sub_dir_num):
        model_data = self.data_model
        sites_id_all = model_data.t_s_dict["sites_id"]
        sites_label_i_index = [j for j in range(len(sites_id_all)) if sites_id_all[j] in sites_ids]
        data_flow = model_data.data_flow[sites_label_i_index, :]
        data_forcing = model_data.data_forcing[sites_label_i_index, :, :]
        data_attr = model_data.data_attr[sites_label_i_index, :]
        stat_dict = {}
        t_s_dict = {}
        source_data_i = copy.deepcopy(model_data.data_source)
        out_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Out"], str(sub_dir_num))
        temp_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Temp"], str(sub_dir_num))
        update_config_item(source_data_i.data_config.data_path, Out=out_dir_new, Temp=temp_dir_new)
        sites_id_all_np = np.array(sites_id_all)
        update_config_item(source_data_i.all_configs, out_dir=out_dir_new, temp_dir=temp_dir_new,
                           flow_screen_gage_id=sites_id_all_np[sites_label_i_index].tolist())
        f_dict = model_data.f_dict
        var_dict = model_data.var_dict
        data_model_i = DataModel(source_data_i, data_flow, data_forcing, data_attr, var_dict, f_dict, stat_dict,
                                 t_s_dict)
        t_s_dict['sites_id'] = sites_id_all_np[sites_label_i_index].tolist()
        t_s_dict['t_final_range'] = source_data_i.t_range
        data_model_i.t_s_dict = t_s_dict
        stat_dict_i = data_model_i.cal_stat_all()
        data_model_i.stat_dict = stat_dict_i
        return data_model_i


class GagesDamDataModel(object):
    def __init__(self, gages_input, nid_input):
        self.gages_input = gages_input
        self.nid_input = nid_input
        self.gage_main_dam_purpose = self.spatial_join_dam()
        self.update_attr()

    def spatial_join_dam(self):
        gage_region_dir = self.gages_input.data_source.all_configs.get("gage_region_dir")
        region_shapefiles = self.gages_input.data_source.all_configs.get("regions")
        # read sites from shapefile of region, get id from it.
        shapefiles = [os.path.join(gage_region_dir, region_shapefile + '.shp') for region_shapefile in
                      region_shapefiles]
        dam_dict = {}
        for shapefile in shapefiles:
            polys = gpd.read_file(shapefile)
            points = self.nid_input.nid_data
            print(points.crs)
            print(polys.crs)
            if not (points.crs == polys.crs):
                points = points.to_crs(polys.crs)
            print(points.head())
            print(polys.head())
            # Make a spatial join
            spatial_dam = gpd.sjoin(points, polys, how="inner", op="within")
            gages_id_dam = spatial_dam['GAGE_ID'].values
            u1 = np.unique(gages_id_dam)
            main_purposes = []
            for u1_i in u1:
                purposes = []
                storages = []
                for index_i in range(gages_id_dam.shape[0]):
                    if gages_id_dam[index_i] == u1_i:
                        purposes.append(spatial_dam["PURPOSES"].iloc[index_i])
                        storages.append(spatial_dam["NID_STORAGE"].iloc[index_i])
                purposes = np.array(purposes)
                storages = np.array(storages)
                u, indices = np.unique(purposes, return_inverse=True)
                max_index = np.amax(indices)
                dict_i = {}
                for i in range(max_index + 1):
                    dict_i[u[i]] = np.sum(storages[np.where(indices == i)])
                main_purpose = max(dict_i.items(), key=operator.itemgetter(1))[0]
                main_purposes.append(main_purpose)
            d = dict(zip(u1.tolist(), main_purposes))
            dam_dict = {**dam_dict, **d}
        return dam_dict

    def update_attr(self):
        dam_dict = self.gage_main_dam_purpose
        attr_lst = self.gages_input.data_source.all_configs.get("attr_chosen")
        data_attr = self.gages_input.data_attr
        stat_dict = self.gages_input.stat_dict
        f_dict = self.gages_input.f_dict
        var_dict = self.gages_input.var_dict
        # update attr_lst, var_dict, f_dict, data_attr
        var_dam = 'GAGE_MAIN_DAM_PURPOSE'
        attr_lst.append(var_dam)
        dam_keys = dam_dict.keys()
        site_dam_purpose = []
        for site_id in self.gages_input.t_s_dict['sites_id']:
            if site_id in dam_keys:
                site_dam_purpose.append(dam_dict[site_id])
            else:
                site_dam_purpose.append(None)
        site_dam_purpose_int, uniques = pd.factorize(site_dam_purpose)
        site_dam_purpose_int = np.array(site_dam_purpose_int).reshape(len(site_dam_purpose_int), 1)
        self.gages_input.data_attr = np.append(data_attr, site_dam_purpose_int, axis=1)
        stat_dict[var_dam] = cal_stat(self.gages_input.data_attr[:, -1])
        # update f_dict and var_dict
        print("update f_dict and var_dict")
        var_dict['dam_purpose'] = var_dam
        f_dict[var_dam] = uniques.tolist()
