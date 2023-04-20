"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-04-20 17:53:21
LastEditors: Wenyu Ouyang
Description: Dicts including models (which are seq-first), losses, and optims
FilePath: /HydroSPDB/hydrospdb/models/model_dict_function.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydrospdb.models.cudnnlstm import (
    CudnnLstmModel,
    LinearCudnnLstmModel,
    KuaiLstm,
)
from hydrospdb.models.lstm_vanilla import CudaLSTM, OfficialLstm
from torch.optim import Adam, SGD, Adadelta
from hydrospdb.models.crits import (
    RMSELoss,
    RmseLoss,
    MultiOutLoss,
    UncertaintyWeights,
    DynamicTaskPrior,
    MultiOutWaterBalanceLoss,
)


"""
Utility dictionaries to map a string to a class.
"""
# now only those models support sequence-first, others are batch-first
sequence_first_model_lst = [
    OfficialLstm,
    CudnnLstmModel,
    KuaiLstm,
    LinearCudnnLstmModel,
]

pytorch_model_dict = {
    "LSTM": OfficialLstm,
    "FreddyLSTM": CudaLSTM,
    "KuaiLSTM": CudnnLstmModel,
    "KuaiLstm": KuaiLstm,
    "KaiTlLSTM": LinearCudnnLstmModel,
}

pytorch_model_wrapper_dict = {}

pytorch_criterion_dict = {
    "RMSE": RMSELoss,
    # xxxSum means that calculate the criterion for each "feature"(the final dim of output), then sum them up
    "RMSESum": RmseLoss,
    "MultiOutLoss": MultiOutLoss,
    "UncertaintyWeights": UncertaintyWeights,
    "DynamicTaskPrior": DynamicTaskPrior,
    "MultiOutWaterBalanceLoss": MultiOutWaterBalanceLoss,
}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "Adadelta": Adadelta}
