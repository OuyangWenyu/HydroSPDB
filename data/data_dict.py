from data.data_camels import Camels
from data.data_gages import Gages
from data.pro.data_gages_pro import GagesPro
from data.data_loaders import BasinFlowDataset, BasinFlowDataModel, XczDataModel, BasinSingleFlowDataset

# more data types which cannot be easily treated same with attribute or forcing data
other_data_source_list = ["RES_STOR_HIST", "GAGES_TS", "FDC", "DI", "WATER_SURFACE"]

datasets_dict = {
    "CAMELS": Camels,
    "GAGES": Gages,
    "GAGES_PRO": GagesPro
}

dataloaders_dict = {
    "StreamflowDataset": BasinFlowDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "StreamflowDataModel": BasinFlowDataModel,
    "KernelFlowDataModel": XczDataModel
}
