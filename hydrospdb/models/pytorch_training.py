"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-04-20 17:55:06
LastEditors: Wenyu Ouyang
Description: Training function for DL models
FilePath: /HydroSPDB/hydrospdb/models/pytorch_training.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from datetime import datetime
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hydrospdb.data.loader.data_loaders import (
    BasinFlowDataModel,
    HydroDlTsDataModel,
)
from hydrospdb.data.loader.data_sets import BasinFlowDataset
from hydrospdb.data.loader.dataloaders4test import TestDataModel
from hydrospdb.models.time_model import PyTorchForecast
from hydrospdb.models.model_dict_function import (
    pytorch_opt_dict,
    pytorch_criterion_dict,
    pytorch_model_wrapper_dict,
    sequence_first_model_lst,
)
from hydrospdb.models.evaluator import generate_predictions
from hydrospdb.models.training_utils import EarlyStopper
from hydrospdb.models.crits import (
    GaussianLoss,
    UncertaintyWeights,
    DynamicTaskPrior,
)
from hydrospdb.utils import hydro_utils
from hydrospdb.utils.hydro_utils import random_index
from hydrospdb.utils.hydro_stat import stat_error


def model_train(forecast_model: PyTorchForecast) -> None:
    """
    Function to train any PyTorchForecast model

    Parameters
    ----------
    forecast_model
        A properly wrapped PyTorchForecast model

    Returns
    -------
    None

    Raises
    -------
    ValueError
        if nan values exist, raise error
    """
    # A dictionary of the necessary parameters for training
    training_params = forecast_model.params["training_params"]
    # The file path to load model weights from; defaults to "model_save"
    model_filepath = forecast_model.params["data_params"]["test_path"]

    es = None
    worker_num = 0
    pin_memory = False
    data_params = forecast_model.params["data_params"]
    num_targets = training_params["multi_targets"]
    if "num_workers" in training_params:
        worker_num = training_params["num_workers"]
        print("using " + str(worker_num) + " workers")
    if "pin_memory" in training_params:
        pin_memory = training_params["pin_memory"]
        print("Pin memory set to " + str(pin_memory))
    if "train_but_not_real" in training_params:
        train_but_not_real = training_params["train_but_not_real"]
    if "early_stopping" in forecast_model.params:
        es = EarlyStopper(forecast_model.params["early_stopping"]["patience"])
    criterion_init_params = {}
    if "criterion_params" in training_params:
        loss_param = training_params["criterion_params"]
        if loss_param is not None:
            for key in loss_param.keys():
                if key == "loss_funcs":
                    criterion_init_params[key] = pytorch_criterion_dict[
                        loss_param[key]
                    ]()
                else:
                    criterion_init_params[key] = loss_param[key]
    if training_params["criterion"] == "MultiOutWaterBalanceLoss":
        # TODO: hard code for streamflow and ET
        stat_dict = forecast_model.training.target_scaler.stat_dict
        stat_dict_keys = list(stat_dict.keys())
        q_name = np.intersect1d(
            [
                "usgsFlow",
                "streamflow",
                "Q",
                "qobs",
            ],
            stat_dict_keys,
        )[0]
        et_name = np.intersect1d(
            [
                "ET",
                "LE",
                "GPP",
                "Ec",
                "Es",
                "Ei",
                "ET_water",
                # sum pf ET components in PML V2
                "ET_sum",
            ],
            stat_dict_keys,
        )[0]
        q_mean = forecast_model.training.target_scaler.stat_dict[q_name][2]
        q_std = forecast_model.training.target_scaler.stat_dict[q_name][3]
        et_mean = forecast_model.training.target_scaler.stat_dict[et_name][2]
        et_std = forecast_model.training.target_scaler.stat_dict[et_name][3]
        means = [q_mean, et_mean]
        stds = [q_std, et_std]
        criterion_init_params["means"] = means
        criterion_init_params["stds"] = stds
    criterion = pytorch_criterion_dict[training_params["criterion"]](
        **criterion_init_params
    )
    params_in_opt = forecast_model.model.parameters()
    if training_params["criterion"] == "UncertaintyWeights":
        # log_var = torch.zeros((1,), requires_grad=True)
        log_vars = [
            torch.zeros((1,), requires_grad=True, device=forecast_model.device)
            for _ in range(training_params["multi_targets"])
        ]
        params_in_opt = [p for p in forecast_model.model.parameters()] + log_vars
    opt = pytorch_opt_dict[training_params["optimizer"]](
        params_in_opt, **training_params["optim_params"]
    )
    max_epochs = training_params["epochs"]
    save_epoch = training_params["save_epoch"]
    save_iter = 0
    if "save_iter" in training_params:
        save_iter = training_params["save_iter"]
    start_epoch = training_params["start_epoch"]
    if issubclass(type(forecast_model.training), Dataset):
        # this means we'll use PyTorch's DataLoader to load the data into batches in each epoch
        data_loader = DataLoader(
            forecast_model.training,
            batch_size=training_params["batch_size"],
            shuffle=True,
            sampler=None,
            batch_sampler=None,
            num_workers=worker_num,
            collate_fn=None,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
        )
        if data_params["t_range_valid"] is not None:
            validation_data_loader = DataLoader(
                forecast_model.validation,
                batch_size=training_params["batch_size"],
                shuffle=False,
                sampler=None,
                batch_sampler=None,
                num_workers=worker_num,
                collate_fn=None,
                pin_memory=pin_memory,
                drop_last=False,
                timeout=0,
                worker_init_fn=None,
            )
    else:
        # use Kuai's method in his WRR paper to iterate https://github.com/mhpi/hydroDL
        data_loader = forecast_model.training
        if data_params["t_range_valid"] is not None:
            validation_data_loader = forecast_model.validation
        # test_data_loader = forecast_model.test_data
        # batch_size * rho must be smaller than ngrid * nt, if not, the value logged will be negative that is wrong
        batch_size = data_params["batch_size"]
        rho = data_params["forecast_history"]
        warmup_length = data_params["warmup_length"]
        ngrid = data_loader.y.shape[0]
        nt = data_loader.y.shape[1]
        while batch_size * rho >= ngrid * nt:
            # try to use a smaller batch_size to make the model runnable
            batch_size = int(batch_size / 10)
        if batch_size < 1:
            batch_size = 1
        n_iter_ep = int(
            np.ceil(
                np.log(0.01)
                / np.log(1 - batch_size * rho / ngrid / (nt - warmup_length))
            )
        )
        assert n_iter_ep >= 1
    session_params = []
    # use tensorboard to visualize the training process
    hyper_param_set = (
        "opt_"
        + training_params["optimizer"]
        + "_lr_"
        + str(opt.defaults["lr"])
        + "_bsize_"
        + str(training_params["batch_size"])
    )
    training_save_dir = os.path.join(model_filepath, hyper_param_set)
    tb = SummaryWriter(training_save_dir)
    param_save_dir = os.path.join(training_save_dir, "training_params")
    if not os.path.exists(param_save_dir):
        os.makedirs(param_save_dir)
    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        if isinstance(data_loader, DataLoader):
            # TODO: don't support MTL models with UncertaintyWeights yet
            total_loss, n_iter_ep = torch_single_train(
                forecast_model.model,
                opt,
                criterion,
                data_loader,
                multi_targets=num_targets,
                device=forecast_model.device,
                writer=tb,
                save_model_iter_dir=param_save_dir,
                save_model_iter=save_iter,
                i_epoch=epoch,
                train_but_not_real=train_but_not_real,
            )
        else:
            if training_params["criterion"] == "UncertaintyWeights":
                total_loss = kuai_single_train(
                    forecast_model.model,
                    opt,
                    criterion,
                    data_loader,
                    forward_params={
                        "n_iter_ep": n_iter_ep,
                        "ngrid": ngrid,
                        "nt": nt,
                        "batch_size": batch_size,
                        "rho": rho,
                        "warmup_length": warmup_length,
                        "device": forecast_model.device,
                        "multi_targets": num_targets,
                    },
                    i_epoch=epoch,
                    save_model_iter_dir=param_save_dir,
                    save_model_iter=save_iter,
                    writer=tb,
                    uw=log_vars,
                    train_but_not_real=train_but_not_real,
                )
            else:
                total_loss = kuai_single_train(
                    forecast_model.model,
                    opt,
                    criterion,
                    data_loader,
                    forward_params={
                        "n_iter_ep": n_iter_ep,
                        "ngrid": ngrid,
                        "nt": nt,
                        "batch_size": batch_size,
                        "rho": rho,
                        "warmup_length": warmup_length,
                        "device": forecast_model.device,
                        "multi_targets": num_targets,
                    },
                    writer=tb,
                    i_epoch=epoch,
                    save_model_iter_dir=param_save_dir,
                    save_model_iter=save_iter,
                    train_but_not_real=train_but_not_real,
                )
        log_str = "Epoch {} Loss {:.3f} time {:.2f}".format(
            epoch, total_loss, time.time() - t0
        )
        tb.add_scalar("Loss", total_loss, epoch)
        print(log_str)
        if data_params["t_range_valid"] is not None:
            # TODO: don't support MTL models with UncertaintyWeights yet
            if isinstance(validation_data_loader, DataLoader):
                valid_obss_np, valid_preds_np, valid_loss = compute_validation(
                    forecast_model.model,
                    criterion,
                    validation_data_loader,
                    device=forecast_model.device,
                )
            else:
                valid_obss_np, valid_preds_np, valid_loss = kuai_compute_validation(
                    forecast_model, criterion, validation_data_loader
                )
            evaluation_metrics = forecast_model.params["evaluate_params"]["metrics"]
            fill_nan = forecast_model.params["evaluate_params"]["fill_nan"]
            target_col = forecast_model.params["data_params"]["target_cols"]
            valid_metrics = evaluate_validation(
                validation_data_loader,
                valid_preds_np,
                valid_obss_np,
                evaluation_metrics,
                fill_nan,
                target_col,
            )
            val_log = "Epoch {} Valid Loss {:.3f} Valid Metric {}".format(
                epoch, valid_loss, valid_metrics
            )
            tb.add_scalar("ValidLoss", valid_loss, epoch)
            for i in range(len(target_col)):
                for evaluation_metric in evaluation_metrics:
                    tb.add_scalar(
                        "Valid" + target_col[i] + evaluation_metric + "mean",
                        np.mean(
                            valid_metrics[evaluation_metric + " of " + target_col[i]]
                        ),
                        epoch,
                    )
                    tb.add_scalar(
                        "Valid" + target_col[i] + evaluation_metric + "median",
                        np.median(
                            valid_metrics[evaluation_metric + " of " + target_col[i]]
                        ),
                        epoch,
                    )
            print(val_log)
            epoch_params = {
                "epoch": epoch,
                "train_loss": str(total_loss),
                "validation_loss": str(valid_loss),
                "validation_metric": valid_metrics,
                "time": log_str,
                "iter_num": n_iter_ep,
            }
            session_params.append(epoch_params)
            if es:
                if not es.check_loss(forecast_model.model, valid_loss):
                    print("Stopping model now")
                    forecast_model.model.load_state_dict(torch.load("checkpoint.pth"))
                    break
        else:
            epoch_params = {
                "epoch": epoch,
                "train_loss": str(total_loss),
                "time": log_str,
                "iter_num": n_iter_ep,
            }
            if training_params["criterion"] == "UncertaintyWeights":
                # hard code for log_vars
                epoch_params = {
                    "epoch": epoch,
                    "train_loss": str(total_loss),
                    "time": log_str,
                    "iter_num": n_iter_ep,
                    "log_vars": str(
                        [(torch.exp(tmp) ** 0.5).item() for tmp in log_vars]
                    ),
                }
            session_params.append(epoch_params)
        if save_epoch > 0 and epoch % save_epoch == 0:
            # save model
            model_file = os.path.join(model_filepath, "model_Ep" + str(epoch) + ".pth")
            save_model(forecast_model.model, model_file)
            # sometimes we train a model in a directory with different hyperparameters
            # we want save models for each of the hyperparameter settings
            model_for_one_training_file = os.path.join(
                param_save_dir, "model_Ep" + str(epoch) + ".pth"
            )
            save_model(forecast_model.model, model_for_one_training_file)
    tb.close()
    forecast_model.params["run"] = session_params
    forecast_model.save_model(model_filepath, max_epochs)
    save_model_params_log(forecast_model.params, training_save_dir)


def save_model_params_log(params, save_log_path):
    params_save_path = os.path.join(
        save_log_path, "params_log_" + str(int(time.time())) + ".json"
    )
    hydro_utils.serialize_json(params, params_save_path)


def save_model(model, model_file):
    try:
        torch.save(model.state_dict(), model_file)
    except:
        torch.save(model.module.state_dict(), model_file)


def evaluate_validation(
    validation_data_loader, output, labels, evaluation_metrics, fill_nan, target_col
):
    """
    calculate metrics for validation

    Parameters
    ----------
    output
        model output
    labels
        model target
    evaluation_metrics
        metrics to evaluate
    fill_nan
        how to fill nan
    target_col
        target columns

    Returns
    -------
    tuple
        metrics
    """
    if type(fill_nan) is list:
        if len(fill_nan) != len(target_col):
            raise Exception("length of fill_nan must be equal to target_col's")
    eval_log = {}
    for i in range(len(target_col)):
        # renormalization to get real metrics
        if type(validation_data_loader.dataset) in [
            BasinFlowDataset,
            BasinFlowDataModel,
        ]:
            # TODO: now only test for BasinFlowDataset
            valid_dataset = TestDataModel(validation_data_loader.dataset)
            pred = valid_dataset.inverse_scale(output[:, :, i : i + 1])
            obs = valid_dataset.inverse_scale(labels[:, :, i : i + 1])
        else:
            pred = output[:, :, i : i + 1]
            obs = labels[:, :, i : i + 1]
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(obs.shape[0], obs.shape[1])
        if type(fill_nan) is str:
            inds = stat_error(obs, pred, fill_nan)
        else:
            inds = stat_error(
                obs,
                pred,
                fill_nan[i],
            )
        for evaluation_metric in evaluation_metrics:
            eval_log[evaluation_metric + " of " + target_col[i]] = inds[
                evaluation_metric
            ].tolist()
    return eval_log


def compute_loss(
    labels: torch.Tensor, output: torch.Tensor, criterion, m: int = 1, **kwargs
) -> float:
    """
    Function for computing the loss

    Parameters
    ----------
    labels
        The real values for the target. Shape can be variable but should follow (batch_size, time)
    output
        The output of the model
    criterion
        loss function
    validation_dataset
        Only passed when unscaling of data is needed.
    m
        defaults to 1
    kwargs
        now specially setting for Uncertainty Weights methods for multi-task models

    Returns
    -------
    float
        the computed loss
    """
    if isinstance(criterion, GaussianLoss):
        if len(output[0].shape) > 2:
            g_loss = GaussianLoss(output[0][:, :, 0], output[1][:, :, 0])
        else:
            g_loss = GaussianLoss(output[0][:, 0], output[1][:, 0])
        loss = g_loss(labels)
        return loss
    if isinstance(output, torch.Tensor):
        if len(labels.shape) != len(output.shape):
            if len(labels.shape) > 1:
                if labels.shape[1] == output.shape[1]:
                    labels = labels.unsqueeze(2)
                else:
                    labels = labels.unsqueeze(0)
    assert labels.shape == output.shape
    if type(criterion) == UncertaintyWeights:
        loss = criterion(output, labels.float(), kwargs["uw"])
    else:
        loss = criterion(output, labels.float())
    return loss


def kuai_single_train(
    model,
    opt,
    criterion,
    data_loader: HydroDlTsDataModel,
    forward_params,
    **kwargs,
):
    """
    Train for one epoch in Kuai Fang's way: http://dx.doi.org/10.1002/2017GL075619

    Parameters
    ----------
    model
        a DL model
    opt
        optimizer
    criterion
        loss function
    data_loader
        a data loader made by ourselves in Kuai's way
    forward_params
        Parameters used in model forwarding if needed
    kwargs
        other options for some special settings, now only for loss computation

    Returns
    -------
    float
        loss of an epoch
    """
    n_iter_ep = forward_params["n_iter_ep"]
    ngrid = forward_params["ngrid"]
    nt = forward_params["nt"]
    batch_size = forward_params["batch_size"]
    rho = forward_params["rho"]
    warmup_length = forward_params["warmup_length"]
    device = forward_params["device"]
    multi_targets = forward_params["multi_targets"]
    loss_ep = 0
    model.train()
    if nt < warmup_length + rho:
        raise ArithmeticError(
            "Please choose a proper warmup_length or rho! nt cannot be smaller than warmup_length + rho"
        )
    i_epoch = kwargs["i_epoch"]
    save_iter = kwargs["save_model_iter"]
    save_dir = kwargs["save_model_iter_dir"]
    writer = kwargs["writer"]
    for i_iter in tqdm(range(0, n_iter_ep), desc="Training Epoch " + str(i_epoch)):
        iter_now = i_iter + 1
        if save_iter > 0 and iter_now % save_iter == 0:
            # save model during training in a epoch
            # iEpoch starts from 1, i_iter starts from 0, we hope both start from 1
            # or iter_now == len(pbar)  # save in the final iter
            model_filepath = os.path.join(
                save_dir,
                "model_epoch_{}_iter_{}.pth".format(i_epoch, iter_now),
            )
            # save_model(model, model_filepath)
            plot_hist_img(model, writer, (i_epoch - 1) * n_iter_ep + iter_now)
        # training iterations
        i_grid, i_t = random_index(ngrid, nt, [batch_size, rho], warmup_length)
        if type(model) is DataParallel:
            if type(model.module) in sequence_first_model_lst:
                batch_first = False
            else:
                batch_first = True
        elif type(model) in list(pytorch_model_wrapper_dict.values()):
            if type(model.model) in sequence_first_model_lst:
                batch_first = False
            else:
                batch_first = True
        else:
            if type(model) in sequence_first_model_lst:
                batch_first = False
            else:
                batch_first = True
        one_batch = data_loader.get_item(
            i_grid, i_t, rho, warmup_length=warmup_length, batch_first=batch_first
        )
        # Convert to CPU/GPU/TPU
        xy = [data_tmp.to(device) for data_tmp in one_batch]
        y_train = xy[-1]
        y_p = model(*xy[0:-1])
        if type(y_p) is tuple:
            others = y_p[1:]
            # Convention: y_p must be the first output of model
            y_p = y_p[0]
        if type(criterion) == DynamicTaskPrior:
            if i_iter == 0:
                kpi = torch.zeros(multi_targets).to(device)
            # "gamma", "kpi_last=None", "alpha"
            loss, kpi = criterion(y_p, y_train.float(), kpi)
        else:
            loss = compute_loss(
                y_train, y_p, criterion=criterion, m=multi_targets, **kwargs
            )
        if torch.isnan(loss) or loss == float("inf"):
            raise ValueError(
                "Error infinite or NaN loss detected. Try normalizing data or performing interpolation or use loss "
                "function which could handle with this case"
            )
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # if we need grad clipping
        opt.step()
        model.zero_grad()
        loss_ep = loss_ep + loss.item()
    if save_iter > 0:
        plot_hist_img(model, writer, i_epoch * n_iter_ep)
    else:
        plot_hist_img(model, writer, i_epoch)
    # print loss
    loss_ep = loss_ep / n_iter_ep
    return loss_ep


def kuai_compute_validation(model, criterion, data_loader: HydroDlTsDataModel):
    """

    Parameters
    ----------
    model
        a PyTorchForecast model
    criterion
        loss function
    data_loader
        a validation data loader made by ourselves in Kuai's way

    Returns
    -------
    float
        loss of an epoch
    """
    model.model.eval()
    data_params = model.params["data_params"]
    device = model.device
    # TODO: not support for all models yet
    # if type(model.test_data) is XarrayDataModel:
    #     test_dataset = TestXarrayDataModel(data_loader)
    #     all_data = test_dataset.load_test_data()
    #     pred = xrds_predictions(
    #         model, test_dataset, *all_data[:-1], device=device, data_params=data_params
    #     )
    if type(data_loader) == DplDataModel:
        valid_dataset = TestDplDataModel(model.training, data_loader)
        all_data = valid_dataset.load_test_data()
        pred_q_et = dpl_model_predictions(
            model, valid_dataset, *all_data, device=device, data_params=data_params
        )
        # only choose streamflow
        pred = pred_q_et[:, :, 0:1]
    else:
        valid_dataset = TestDataModel(data_loader)
        all_data = valid_dataset.load_test_data()
        pred = generate_predictions(
            model, valid_dataset, *all_data[:-1], device=device, data_params=data_params
        )
    if type(model.model) is DataParallel:
        if type(model.model.module) in sequence_first_model_lst:
            seq_first = True
        else:
            seq_first = False
    else:
        if type(model.model) in sequence_first_model_lst:
            seq_first = True
        else:
            seq_first = False
    if seq_first:
        valid_loss = compute_loss(
            torch.Tensor(all_data[-1].swapaxes(0, 1)),
            torch.Tensor(pred.swapaxes(0, 1)),
            criterion,
        )
    else:
        valid_loss = compute_loss(
            torch.Tensor(all_data[-1]),
            torch.Tensor(pred),
            criterion,
        )

    if pred.shape[1] != all_data[-1].shape[1]:
        # it means we use an Nto1 mode model, so cutoff some previous data for observations to be comparable
        return (
            all_data[-1][:, valid_dataset.test_data.rho - 1 :, :],
            pred,
            valid_loss,
        )
    return all_data[-1], pred, valid_loss


def torch_single_train(
    model,
    opt: optim.Optimizer,
    criterion,
    data_loader: DataLoader,
    multi_targets=1,
    device=None,
    **kwargs,
) -> float:
    """
    Training function for one epoch

    Parameters
    ----------
    model
        a PyTorch model inherit from nn.Module
    opt
        optimizer function from PyTorch optim.Optimizer
    criterion
        loss function
    data_loader
        object for loading data to the model
    multi_targets
        with multi targets, we will use different loss function
    device
        where we put the tensors and models

    Returns
    -------
    tuple(float, int)
        loss of this epoch and number of all iterations

    Raises
    --------
    ValueError
        if nan exits, raise a ValueError
    """
    # we will set model.eval() in the validation function so here we should set model.train()
    model.train()
    writer = kwargs["writer"]
    n_iter_ep = 0
    running_loss = 0.0
    if type(model) is DataParallel:
        if type(model.module) in sequence_first_model_lst:
            seq_first = True
        else:
            seq_first = False
    elif type(model) in list(pytorch_model_wrapper_dict.values()):
        if type(model.model) in sequence_first_model_lst:
            seq_first = True
        else:
            seq_first = False
    else:
        if type(model) in sequence_first_model_lst:
            seq_first = True
        else:
            seq_first = False
    pbar = tqdm(data_loader)
    i_epoch = kwargs["i_epoch"]
    save_iter = kwargs["save_model_iter"]
    save_dir = kwargs["save_model_iter_dir"]

    train_but_not_real = kwargs["train_but_not_real"]
    if train_but_not_real:
        weight_path = os.path.join(save_dir, "model_Ep{}.pth".format(i_epoch))
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        plot_hist_img(model, writer, i_epoch)
        # no train, no loss, give it -999
        return -999, len(data_loader)
    for i, (src, trg) in enumerate(pbar):
        # iEpoch starts from 1, iIter starts from 0, we hope both start from 1
        iter_now = i + 1
        if save_iter > 0 and iter_now % save_iter == 0:
            # save model during training in a epoch
            # or iter_now == len(pbar)  # save in the final iter
            model_filepath = os.path.join(
                save_dir,
                "model_epoch_{}_iter_{}.pth".format(i_epoch, iter_now),
            )
            # save_model(model, model_filepath)
            plot_hist_img(model, writer, (i_epoch - 1) * len(pbar) + iter_now)
        # Convert to CPU/GPU/TPU
        if type(src) is list:
            xs = [
                data_tmp.permute([1, 0, 2]).to(device)
                if seq_first and data_tmp.ndim == 3
                else data_tmp.to(device)
                for data_tmp in src
            ]
        else:
            xs = [
                src.permute([1, 0, 2]).to(device)
                if seq_first and src.ndim == 3
                else src.to(device)
            ]
        trg = (
            trg.permute([1, 0, 2]).to(device)
            if seq_first and trg.ndim == 3
            else trg.to(device)
        )
        output = model(*xs)
        if type(output) is tuple:
            others = output[1:]
            # Convention: y_p must be the first output of model
            output = output[0]
        loss = compute_loss(trg, output, criterion, m=multi_targets, **kwargs)
        if loss > 100:
            print("Warning: high loss detected")
        loss.backward()
        opt.step()
        model.zero_grad()
        if torch.isnan(loss) or loss == float("inf"):
            raise ValueError(
                "Error infinite or NaN loss detected. Try normalizing data or performing interpolation"
            )
        running_loss += loss.item()
        n_iter_ep += 1
    if save_iter > 0:
        plot_hist_img(model, writer, i_epoch * n_iter_ep)
    else:
        plot_hist_img(model, writer, i_epoch)
    total_loss = running_loss / float(n_iter_ep)
    return total_loss, n_iter_ep


def plot_hist_img(model, writer, global_step):
    for tag, parm in model.named_parameters():
        writer.add_histogram(tag + "_hist", parm.detach().cpu().numpy(), global_step)
        if len(parm.shape) == 2:
            img_format = "HW"
            if parm.shape[0] > parm.shape[1]:
                img_format = "WH"
            writer.add_image(
                tag + "_img",
                parm.detach().cpu().numpy(),
                global_step,
                dataformats=img_format,
            )


def compute_validation(
    model,
    criterion,
    data_loader: DataLoader,
    device: torch.device = None,
) -> float:
    """
    Function to compute the validation loss metrics

    Parameters
    ----------
    model
        the trained model
    criterion
        torch.nn.modules.loss
    dataloader
        The data-loader of either validation or test-data
    device
        torch.device

    Returns
    -------
    tuple
        validation observations (numpy array), predictions (numpy array) and the loss of validation
    """
    # TODO: not fully support dPL model yet, only support dpl-ann and dpl-lstm models' final-mode computation now, else are not tested
    model.eval()
    obs = []
    preds = []
    if type(model) is DataParallel:
        if type(model.module) in sequence_first_model_lst:
            seq_first = True
        else:
            seq_first = False
    else:
        if type(model) in sequence_first_model_lst:
            seq_first = True
        else:
            seq_first = False
    if seq_first:
        cat_dim = 1
    else:
        cat_dim = 0
    with torch.no_grad():
        for src, trg in data_loader:
            if type(src) is list:
                xs = [
                    data_tmp.permute([1, 0, 2]).to(device)
                    if seq_first and data_tmp.ndim == 3
                    else data_tmp.to(device)
                    for data_tmp in src
                ]
            else:
                xs = [
                    src.permute([1, 0, 2]).to(device)
                    if seq_first and src.ndim == 3
                    else src.to(device)
                ]
            trg = (
                trg.permute([1, 0, 2]).to(device)
                if seq_first and trg.ndim == 3
                else trg.to(device)
            )
            output = model(*xs)
            if type(output) is tuple:
                others = output[1:]
                # Convention: y_p must be the first output of model
                output = output[0]
            obs.append(trg)
            preds.append(output)
        obs_final = torch.cat(obs, dim=cat_dim)
        pred_final = torch.cat(preds, dim=cat_dim)
        valid_loss = compute_loss(obs_final, pred_final, criterion)
    if seq_first:
        y_obs = obs_final.detach().cpu().numpy().swapaxes(0, 1)
        y_pred = pred_final.detach().cpu().numpy().swapaxes(0, 1)
    else:
        y_obs = obs_final.detach().cpu().numpy()
        y_pred = pred_final.detach().cpu().numpy()
    return y_obs, y_pred, valid_loss
