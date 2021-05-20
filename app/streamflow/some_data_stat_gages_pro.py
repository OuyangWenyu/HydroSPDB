import csv
import os
import pandas as pd
import definitions
from data.pro.data_gages_pro import GagesPro, get_max_dam_norm_stor, get_dam_storage_std

gage_2909id_file = os.path.join(definitions.ROOT_DIR, "example", "2909basins_NSE.csv")
sites_id = pd.read_csv(gage_2909id_file, dtype={0: str}).iloc[:, 0].values

gages_dir = [os.path.join(definitions.DATASET_DIR, "gages_pro"),
             os.path.join(definitions.DATASET_DIR, "gages"),
             os.path.join(definitions.DATASET_DIR, "nid"),
             os.path.join(definitions.DATASET_DIR, "gridmet")]
gages = GagesPro(gages_dir, download=False)
max_norm_stors = get_max_dam_norm_stor(gages, sites_id)
std_norm_stor = get_dam_storage_std(gages, sites_id)
attr_lst = ["MAX_NORMAL_STOR", "DAM_STORAGE_STD", "DOR"]
data_dor = gages.read_constant_cols(sites_id, [attr_lst[2]])
df = pd.DataFrame(
    {"GAGE_ID": sites_id, attr_lst[0] + " (Unit:Acre-Feet)": max_norm_stors, attr_lst[1]: std_norm_stor,
     attr_lst[2] + " (source data from GAGES-II)": data_dor[:, 0]})
df.to_csv(os.path.join(definitions.ROOT_DIR, "example", "gages", "exp2",
                       '2909basins_MAX-NORMAL-STOR_DAM-STORAGE-STD_DOR.csv'), quoting=csv.QUOTE_NONNUMERIC, index=None)
