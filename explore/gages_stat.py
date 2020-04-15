import os
import numpy as np
import geopandas as gpd
import pandas as pd

from data.data_input import load_result
from explore.stat import statError


def stat_every_region(gages_data_model, epoch):
    id_regions_idx, id_regions_sites_ids = ids_of_regions(gages_data_model)
    preds, obss, inds_dfs = split_results_to_regions(gages_data_model, epoch, id_regions_idx, id_regions_sites_ids)
    region_names = gages_data_model.data_source.all_configs.get("regions")
    inds_medians = []
    inds_means = []
    for i in range(len(region_names)):
        inds_medians.append(inds_dfs[i].median(axis=0))
        inds_means.append(inds_dfs[i].mean(axis=0))
    return inds_medians, inds_means


def ids_of_regions(gages_data_model):
    gage_region_dir = gages_data_model.data_source.all_configs.get("gage_region_dir")
    region_shapefiles = gages_data_model.data_source.all_configs.get("regions")
    shapefiles = [os.path.join(gage_region_dir, region_shapefile + '.shp') for region_shapefile in
                  region_shapefiles]
    df_id_region = np.array(gages_data_model.t_s_dict["sites_id"])
    assert (all(x < y for x, y in zip(df_id_region, df_id_region[1:])))
    id_regions_idx = []
    id_regions_sites_ids = []
    for shapefile in shapefiles:
        shape_data = gpd.read_file(shapefile)
        gages_id = shape_data['GAGE_ID'].values
        c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
        assert (all(x < y for x, y in zip(ind1, ind1[1:])))
        assert (all(x < y for x, y in zip(c, c[1:])))
        id_regions_idx.append(ind1)
        id_regions_sites_ids.append(c)
    return id_regions_idx, id_regions_sites_ids


def split_results_to_regions(gages_data_model, epoch, id_regions_idx, id_regions_sites_ids):
    regions_name = gages_data_model.data_source.all_configs.get("regions")
    pred_all, obs_all = load_result(gages_data_model.data_source.data_config.data_path['Temp'], epoch)
    pred_all = pred_all.reshape(pred_all.shape[0], pred_all.shape[1])
    obs_all = obs_all.reshape(obs_all.shape[0], obs_all.shape[1])
    preds = []
    obss = []
    inds_dfs = []
    for i in range(len(id_regions_idx)):
        pred = pred_all[id_regions_idx[i], :]
        obs = obs_all[id_regions_idx[i], :]
        preds.append(pred)
        obss.append(obs)
        inds = statError(obs, pred)
        inds['STAID'] = id_regions_sites_ids[i]
        inds_df = pd.DataFrame(inds)
        # inds_df.to_csv(os.path.join(gages_data_model.data_source.data_config.data_path["Out"],
        #                             regions_name[i] + "epoch" + str(epoch) + 'data_df.csv'))
        inds_dfs.append(inds_df)
    return preds, obss, inds_dfs
