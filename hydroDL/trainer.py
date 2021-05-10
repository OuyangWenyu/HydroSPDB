import os
import random
import numpy as np
from typing import Dict
import pandas as pd
import torch

from data.data_dict import datasets_dict
from explore.stat import statError
from hydroDL.evaluator import evaluate_model
from hydroDL.pytorch_training import model_train
from hydroDL.time_model import PyTorchForecast
from utils.hydro_utils import serialize_numpy, unserialize_numpy


def set_random_seed(seed):
    print("Random seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate(params: Dict):
    """Function to train and test a Model.
    :param params: Dictionary containing all the parameters needed to run the model
    :type Dict:
    """
    random_seed = params["training_params"]["random_seed"]
    set_random_seed(random_seed)
    dataset_params = params["dataset_params"]
    dataset_name = dataset_params["dataset_name"]
    dataset = datasets_dict[dataset_name](dataset_params["data_path"], dataset_params["download"])
    model = PyTorchForecast(params["model_params"]["model_name"], dataset, params)
    training_params = params["training_params"]
    model_file_path = dataset_params["test_path"]
    if params["training_params"]["train_mode"]:
        if ("weight_path" in params["model_params"] and params["model_params"]["continue_train"]) or (
                "weight_path" not in params["model_params"]):
            model_train(model, training_params, model_filepath=model_file_path)
        model_type = params["model_params"]["model_type"]
        test_acc = evaluate_model(model, model_type, params["dataset_params"]["target_cols"], params["metrics"],
                                  params["dataset_params"], {})
        print("summary test_accuracy", test_acc[0])
        # save the results
        save_result(dataset_params['test_path'], params["training_params"]["epochs"], test_acc[1], test_acc[2])
    # TODO: plots using matplotlib/seaborn/plotly/cartopy


def save_result(save_dir, epoch, pred, obs, pred_name='flow_pred', obs_name='flow_obs'):
    """save the pred value of testing period and obs value"""
    flow_pred_file = os.path.join(save_dir, 'epoch' + str(epoch) + pred_name)
    flow_obs_file = os.path.join(save_dir, 'epoch' + str(epoch) + obs_name)
    serialize_numpy(pred, flow_pred_file)
    serialize_numpy(obs, flow_obs_file)


def load_result(save_dir, epoch, pred_name='flow_pred', obs_name='flow_obs') -> (np.array, np.array):
    """load the pred value of testing period and obs value"""
    flow_pred_file = os.path.join(save_dir, 'epoch' + str(epoch) + pred_name + '.npy')
    flow_obs_file = os.path.join(save_dir, 'epoch' + str(epoch) + obs_name + '.npy')
    pred = unserialize_numpy(flow_pred_file)
    obs = unserialize_numpy(flow_obs_file)
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(obs.shape[0], obs.shape[1])
    return pred, obs


def stat_result(save_dirs, test_epoch, return_value=False) -> (pd.DataFrame, np.array, np.array):
    pred, obs = load_result(save_dirs, test_epoch)
    inds = statError(obs, pred)
    inds_df = pd.DataFrame(inds)
    if return_value:
        return inds_df, pred, obs
    return inds_df


def load_ensemble_result(save_dirs, test_epoch) -> (pd.DataFrame, np.array, np.array):
    preds = []
    obss = []
    for save_dir in save_dirs:
        pred_i, obs_i = load_result(save_dir, test_epoch)
        if pred_i.ndim == 3 and pred_i.shape[-1] == 1:
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
        preds.append(pred_i)
        obss.append(obs_i)
    preds_np = np.array(preds)
    obss_np = np.array(obss)
    pred_mean = np.mean(preds_np, axis=0)
    obs_mean = np.mean(obss_np, axis=0)
    return pred_mean, obs_mean


def stat_ensemble_result(save_dirs, test_epoch, return_value=False) -> (np.array, np.array):
    pred_mean, obs_mean = load_ensemble_result(save_dirs, test_epoch)
    inds = statError(obs_mean, pred_mean)
    inds_df = pd.DataFrame(inds)
    if return_value:
        return inds_df, pred_mean, obs_mean
    return inds_df
