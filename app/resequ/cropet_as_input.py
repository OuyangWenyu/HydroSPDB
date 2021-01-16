import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import torch

from data import GagesConfig
from data.data_input import GagesModel
from data.gages_input_dataset import GagesEtDataModel, GagesModels
from data.gridmet_input import GridmetConfig, GridmetSource, GridmetModel
from hydroDL.master.master import master_train_gridmet

sys.path.append("../..")
import definitions
from utils import unserialize_json_ordered
from data.config import cfg

# Firstly, choose the dammed basins with irrigation as the main purpose of reservoirs
# prerequisite: app/streamflow/gages_conus_result_section2.py has been run
nid_dir = os.path.join(cfg.NID.NID_DIR, "test")
main_purpose_file = os.path.join(nid_dir, "dam_main_purpose_dict.json")
all_sites_purposes_dict = unserialize_json_ordered(main_purpose_file)
all_sites_purposes = pd.Series(all_sites_purposes_dict)
include_irr = all_sites_purposes.apply(lambda x: "I" in x)
irri_basins_id = all_sites_purposes[include_irr == True].index.tolist()
df = pd.DataFrame({"GAGE_ID": irri_basins_id})

OUTPUT_IRRI_GAGE_ID = False
if OUTPUT_IRRI_GAGE_ID:
    df.to_csv("irrigation_gage_id.csv", index=None)

# the major procedures for only 300+ irrigation basins:
# 1. weighted average value of all crops in a basin for ETc
# 2. use the forcing data from gridmet as the input to LSTM
# 3. add ETo as additional input
# 4. use ETo and ETc as additional inputs
# 5. compare the CDFs of all these 3 cases
t_range_train = ["2008-01-01", "2013-01-01"]
t_range_test = ["2013-01-01", "2018-01-01"]

config4gridmet = copy.deepcopy(cfg)
config4gridmet.MODEL.tRangeTrain = t_range_train
config4gridmet.MODEL.tRangeTest = t_range_test
config4gridmet.GAGES.streamflowScreenParams = {'missing_data_ratio': 1, 'zero_value_ratio': 1}
config_data = GagesConfig(config4gridmet)
gages_model = GagesModels(config_data, screen_basin_area_huc4=False, sites_id=irri_basins_id)
gages_model_train = gages_model.data_model_train
gages_model_test = gages_model.data_model_test

CROP_ET_ZIP_DIR = os.path.join(definitions.ROOT_DIR, "example", "data", "gridmet")
gridmet_config = GridmetConfig(CROP_ET_ZIP_DIR)
gridmet_source = GridmetSource(gridmet_config, irri_basins_id)
gridmet_data_model_train = GridmetModel(gridmet_source, t_range_train)
gridmet_data_model_test = GridmetModel(gridmet_source, t_range_test, is_test=True,
                                       stat_train=gridmet_data_model_train.stat_forcing_dict)

with torch.cuda.device(0):
    data_et_model = GagesEtDataModel(gages_model_train, gridmet_data_model_train, True)
    master_train_gridmet(data_et_model)
#################################################
# The second method for these 300+ basins
# 1. use the forcing data from gridmet as the input to LSTM-CONUS (3557 basins)
# 2. retrain the model only using data of the 300+ basins
# (use the weights of LSTM-CONUS as inital weights, then train on  300+ basins)
# 3. use ETo as additonal input
# 4. weighted average value of all crops in a basin for ETc and use ETo and ETc as additional inputs
# 5. compare the CDFs of all these 3 cases

#################################################
# the major procedures for 3557 basins:
# 1. weighted average value of all crops in a basin for ETc
# 2. use the forcing data from gridmet as the input to LSTM
# 3. add ETo as additional input
# 4. use ETo and ETc as additional inputs
# 5. compare the CDFs of all these 3 cases
# 6. compare the 300+ basins of the 3557 cases
