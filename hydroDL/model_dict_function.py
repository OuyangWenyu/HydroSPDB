from hydroDL.basic.cudnnlstm import CudnnLstmModel, LinearCudnnLstmModel, CNN1dLCmodel, CudnnLstmModelLstmKernel
from hydroDL.transformer_xl.multi_head_base import MultiAttnHeadSimple
from hydroDL.transformer_xl.transformer_basic import SimpleTransformer, CustomTransformerDecoder
from hydroDL.transformer_xl.informer import Informer
from hydroDL.transformer_xl.transformer_xl import TransformerXL
from hydroDL.transformer_xl.dummy_torch import DummyTorchModel
from hydroDL.basic.linear_regression import SimpleLinearModel
from hydroDL.basic.lstm_vanilla import LSTMForecast, CudaLSTM
from torch.optim import Adam, SGD, Adadelta
from torch.nn import MSELoss, SmoothL1Loss, PoissonNLLLoss, L1Loss
from hydroDL.custom.custom_opt import BertAdam, RmseLoss
from hydroDL.basic.linear_regression import simple_decode
from hydroDL.transformer_xl.transformer_basic import greedy_decode
from hydroDL.da_rnn.model import DARNN
from hydroDL.custom.custom_opt import (RMSELoss, MAPELoss, PenalizedMSELoss, NegativeLogLikelihood, MASELoss,
                                       GaussianLoss)
from hydroDL.transformer_xl.transformer_bottleneck import DecoderTransformer
from hydroDL.custom.dilate_loss import DilateLoss
from hydroDL.meta_models.basic_ae import AE
import torch

"""
Utility dictionaries to map a string to a class.
"""
# TODO: now only those models support sequence-first, others are batch-first
sequence_first_model_lst = [CudnnLstmModel, LinearCudnnLstmModel, CNN1dLCmodel, CudnnLstmModelLstmKernel]

pytorch_model_dict = {
    "LSTM": LSTMForecast,
    "FreddyLSTM": CudaLSTM,
    "KuaiLSTM": CudnnLstmModel,
    "KaiTlLSTM": LinearCudnnLstmModel,
    "DapengCNNLSTM": CNN1dLCmodel,
    "LSTMKernel": CudnnLstmModelLstmKernel,
    "SimpleLinearModel": SimpleLinearModel,
    "DARNN": DARNN,
    # the above are tested, the following are not
    "CustomTransformerDecoder": CustomTransformerDecoder,
    "MultiAttnHeadSimple": MultiAttnHeadSimple,
    "SimpleTransformer": SimpleTransformer,
    "TransformerXL": TransformerXL,
    "DummyTorchModel": DummyTorchModel,
    "DecoderTransformer": DecoderTransformer,
    "BasicAE": AE,
    "Informer": Informer
}

pytorch_criterion_dict = {
    "GaussianLoss": GaussianLoss,
    "MASELoss": MASELoss,
    "MSE": MSELoss,
    "SmoothL1Loss": SmoothL1Loss,
    "PoissonNLLLoss": PoissonNLLLoss,
    "RMSE": RMSELoss,
    # xxxSum means that calculate the criterion for each "feature"(the final dim of output), then sum them up
    "RMSESum": RmseLoss,
    "MAPE": MAPELoss,
    "DilateLoss": DilateLoss,
    "L1": L1Loss,
    "PenalizedMSELoss": PenalizedMSELoss,
    "NegativeLogLikelihood": NegativeLogLikelihood}

decoding_functions = {"greedy_decode": greedy_decode, "simple_decode": simple_decode}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "BertAdam": BertAdam, "Adadelta": Adadelta}


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
