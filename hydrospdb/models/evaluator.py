"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-04-20 17:52:58
LastEditors: Wenyu Ouyang
Description: Testing functions for hydroDL models
FilePath: /HydroSPDB/hydrospdb/models/evaluator.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
from typing import Dict, Tuple
from functools import reduce
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from hydrospdb.data.loader.dataloaders4test import (
    TestDataModel,
)
from hydrospdb.utils.hydro_stat import stat_error
from hydrospdb.utils import hydro_utils

from hydrospdb.models.model_dict_function import sequence_first_model_lst
from hydrospdb.models.time_model import PyTorchForecast
from hydrospdb.models.training_utils import get_the_device


def evaluate_model(model: PyTorchForecast) -> Tuple[Dict, np.array, np.array]:
    """
    A function to evaluate a model, called at end of training.

    Parameters
    ----------
    model
        the DL model class

    Returns
    -------
    tuple[dict, np.array, np.array]
        eval_log, denormalized predictions and observations
    """
    data_params = model.params["data_params"]
    # types of observations
    target_col = model.params["data_params"]["target_cols"]
    evaluation_metrics = model.params["evaluate_params"]["metrics"]
    # fill_nan: "no" means ignoring the NaN value;
    #           "sum" means calculate the sum of the following values in the NaN locations.
    #           For example, observations are [1, nan, nan, 2], and predictions are [0.3, 0.3, 0.3, 1.5].
    #           Then, "no" means [1, 2] v.s. [0.3, 1.5] while "sum" means [1, 2] v.s. [0.3 + 0.3 + 0.3, 1.5].
    #           If it is a str, then all target vars use same fill_nan method;
    #           elif it is a list, each for a var
    fill_nan = model.params["evaluate_params"]["fill_nan"]
    # save result here
    eval_log = {}

    # test the trained model
    test_epoch = model.params["evaluate_params"]["test_epoch"]
    train_epoch = model.params["training_params"]["epochs"]
    if test_epoch != train_epoch:
        # Generally we use same epoch for train and test, but sometimes not
        # TODO: better refactor this part, because sometimes we save multi models for multi hyperparameters
        model_filepath = model.params["data_params"]["test_path"]
        model.model = model.load_model(
            model.params["model_params"]["model_name"],
            model.params["model_params"],
            weight_path=os.path.join(model_filepath, f"model_Ep{str(test_epoch)}.pth"),
        )
    pred, obs, test_data = infer_on_torch_model(model)
    print("Un-transforming data")
    preds_np = test_data.inverse_scale(pred)
    obss_np = test_data.inverse_scale(obs)

    #  Then evaluate the model metrics
    if type(fill_nan) is list and len(fill_nan) != len(target_col):
        raise ArithmeticError("length of fill_nan must be equal to target_col's")
    for i in range(len(target_col)):
        if type(fill_nan) is str:
            inds = stat_error(obss_np[:, :, i], preds_np[:, :, i], fill_nan)
        else:
            inds = stat_error(obss_np[:, :, i], preds_np[:, :, i], fill_nan[i])
        for evaluation_metric in evaluation_metrics:
            eval_log[evaluation_metric + " of " + target_col[i]] = inds[
                evaluation_metric
            ]

    return eval_log, preds_np, obss_np


def infer_on_torch_model(
    model: PyTorchForecast,
) -> Tuple[torch.Tensor, torch.Tensor, TestDataModel]:
    """
    Function to handle both test evaluation and inference on a test data-frame.
    """
    data_params = model.params["data_params"]
    device = get_the_device(model.params["training_params"]["device"])
    model.model.eval()
    test_dataset = TestDataModel(model.test_data)
    all_data = test_dataset.load_test_data()
    pred = generate_predictions(
        model, test_dataset, *all_data[:-1], device=device, data_params=data_params
    )
    if pred.shape[1] != all_data[-1].shape[1]:
        # it means we use an Nto1 mode model, so cutoff some previous data for observations to be comparable
        return pred, all_data[-1][:, test_dataset.test_data.rho - 1 :, :], test_dataset
    return pred, all_data[-1], test_dataset


def generate_predictions(
    ts_model: PyTorchForecast,
    test_model: TestDataModel,
    *args,
    device: torch.device,
    data_params: dict,
    return_cell_state: bool = False,
) -> np.ndarray:
    """Perform Evaluation on the test (or valid) data.

    Parameters
    ----------
    ts_model : PyTorchForecast
        _description_
    test_model : TestDataModel
        _description_
    device : torch.device
        _description_
    data_params : dict
        _description_
    return_cell_state : bool, optional
        if True, time-loop evaluation for cell states, by default False
        NOTE: ONLY for LSTM models

    Returns
    -------
    np.ndarray
        _description_
    """
    model = ts_model.model
    model.train(mode=False)
    seq_first = type(model) in sequence_first_model_lst
    if issubclass(type(test_model.test_data), Dataset):
        # TODO: not support return_cell_states yet
        # here the batch is just an index of lookup table, so any batch size could be chosen
        test_loader = DataLoader(
            test_model.test_data, batch_size=data_params["batch_size"], shuffle=False
        )
        test_preds = []
        with torch.no_grad():
            for i_batch, (xs, ys) in enumerate(test_loader):
                # here the a batch doesn't mean a basin; it is only an index in lookup table
                # for NtoN mode, only basin is index in lookup table, so the batch is same as basin
                # for Nto1 mode, batch is only an index
                if seq_first:
                    xs = xs.transpose(0, 1)
                    ys = ys.transpose(0, 1)
                xs = xs.to(device)
                ys = ys.to(device)
                output = model(xs)
                if type(output) is tuple:
                    others = output[1:]
                    # Convention: y_p must be the first output of model
                    output = output[0]
                if seq_first:
                    output = output.transpose(0, 1)
                test_preds.append(output.cpu().numpy())
            pred = reduce(lambda x, y: np.vstack((x, y)), test_preds)
        if pred.ndim == 2:
            # the ndim is 2 meaning we use an Nto1 mode
            # as lookup table is (basin 1's all time length, basin 2's all time length, ...)
            # params of reshape should be (basin size, time length)
            pred = pred.flatten().reshape(test_model.test_data.y.shape[0], -1, 1)

    else:
        x = args[0]
        c = args[1]
        z = None
        if len(args) == 3:
            z = args[2]
        ngrid, nt, nx = x.shape
        if c is not None:
            nc = c.shape[-1]

        i_s = np.arange(0, ngrid, data_params["batch_size"])
        i_e = np.append(i_s[1:], ngrid)

        y_out_list = []
        if return_cell_state:
            # all basins' cell states
            cs_out_lst = []
        for i in range(len(i_s)):
            # print("batch {}".format(i))
            x_temp = x[i_s[i] : i_e[i], :, :]

            if c is not None and c.shape[-1] > 0:
                c_temp = np.repeat(
                    np.reshape(c[i_s[i] : i_e[i], :], [i_e[i] - i_s[i], 1, nc]),
                    nt,
                    axis=1,
                )
                xhTest = (
                    torch.from_numpy(
                        np.swapaxes(np.concatenate([x_temp, c_temp], 2), 1, 0)
                    ).float()
                    if seq_first
                    else torch.from_numpy(
                        np.concatenate([x_temp, c_temp], 2)
                    ).float()
                )
            elif seq_first:
                xhTest = torch.from_numpy(np.swapaxes(x_temp, 1, 0)).float()
            else:
                xhTest = torch.from_numpy(x_temp).float()
            xhTest = xhTest.to(device)
            with torch.no_grad():
                if z is not None:
                    # now only support z is 2d var
                    assert z.ndim == 2
                    if seq_first:
                        zTemp = z[i_s[i] : i_e[i], :]
                        zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
                    else:
                        zTest = torch.from_numpy(z[i_s[i] : i_e[i], :]).float()
                    zTest = zTest.to(device)
                    y_p = model(xhTest, zTest)
                else:
                    if return_cell_state:
                        cs_lst = []
                        for j in range(nt):
                            y_p_, (hs, cs) = model(
                                xhTest[0 : j + 1, :, :], return_h_c=True
                            )
                            cs_lst.append(cs)
                        cs_cat_lst = torch.cat(cs_lst, dim=0)
                    y_p = model(xhTest)
                if type(y_p) is tuple:
                    others = y_p[1:]
                    # Convention: y_p must be the first output of model
                    y_p = y_p[0]
                if seq_first:
                    y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
                else:
                    y_out = y_p.detach().cpu().numpy()

                y_out_list.append(y_out)
                if return_cell_state:
                    if seq_first:
                        cs_out = cs_cat_lst.detach().cpu().numpy().swapaxes(0, 1)
                    else:
                        cs_out = cs_cat_lst.detach().cpu().numpy()
                    cs_out_lst.append(cs_out)
        # model.zero_grad()
        torch.cuda.empty_cache()
        pred = reduce(lambda a, b: np.vstack((a, b)), y_out_list)
        if return_cell_state:
            cell_state = reduce(lambda a, b: np.vstack((a, b)), cs_out_lst)
            np.save(
                os.path.join(data_params["test_path"], "cell_states.npy"), cell_state
            )
            return pred, cell_state

    return pred
