import csv
import os
import pandas as pd
import definitions
from data.data_gages import Gages
from utils.hydro_utils import unserialize_json

gage_3557id_file = os.path.join(definitions.ROOT_DIR, "example", "3557basins_ID_NSE_DOR.csv")
sites_id = pd.read_csv(gage_3557id_file, dtype={0: str}).iloc[:, 0].values

cfg_dir_gages = os.path.join(definitions.ROOT_DIR, "example", "gages")
cfg_file = os.path.join(cfg_dir_gages, "exp2", "05_April_202105_47AM.json")
config_data = unserialize_json(cfg_file)

gages = Gages(config_data["dataset_params"]["data_path"], download=False)
attr_lst = ["MAJ_NDAMS_2009", "RAW_DIS_NEAREST_DAM", "RAW_AVG_DIS_ALL_MAJ_DAMS"]
data_attr = gages.read_constant_cols(sites_id, attr_lst)
df = pd.DataFrame(
    {"GAGE_ID": sites_id, attr_lst[0]: data_attr[:, 0], attr_lst[1]: data_attr[:, 1], attr_lst[2]: data_attr[:, 2]})
df.to_csv(os.path.join(config_data["dataset_params"]["test_path"], '3557basins_NDAMS_NEAREST_AVGDIS.csv'),
          quoting=csv.QUOTE_NONNUMERIC, index=None)
