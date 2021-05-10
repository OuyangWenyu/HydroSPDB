import os
import time

import numpy as np
import torch
import torch.optim as optim
from typing import Type, Dict
from torch.utils.data import Dataset, DataLoader

from data.data_loaders import HydroDlTsDataModel
from hydroDL.time_model import PyTorchForecast
from hydroDL.model_dict_function import pytorch_opt_dict, pytorch_criterion_dict, sequence_first_model_lst
from hydroDL.transformer_xl.transformer_basic import greedy_decode
from hydroDL.basic.linear_regression import simple_decode
from hydroDL.training_utils import EarlyStopper
from hydroDL.custom.custom_opt import GaussianLoss, MASELoss
from utils.hydro_utils import numpy_to_tvar, random_index


def model_train(forecast_model: PyTorchForecast,
                training_params: Dict,
                takes_target=False,
                forward_params: Dict = {},
                model_filepath: str = "model_save") -> None:
    """Function to train any PyTorchForecast model

    :param forecast_model:  A properly wrapped PyTorchForecast model
    :type forecast_model: PyTorchForecast
    :param training_params: A dictionary of the necessary parameters for training.
    :type training_params: Dict
    :param takes_target: A parameter to determine whether a model requires the target, defaults to False
    :type takes_target: bool, optional
    :param forward_params: [description], defaults to {}
    :type forward_params: Dict, optional
    :param model_filepath: The file path to load modeel weights from, defaults to "model_save"
    :type model_filepath: str, optional
    :raises ValueError: [description]
    """
    es = None
    worker_num = 1
    pin_memory = False
    dataset_params = forecast_model.params["dataset_params"]
    num_targets = 1
    if "n_targets" in forecast_model.params:
        num_targets = forecast_model.params["n_targets"]
    if "num_workers" in dataset_params:
        worker_num = dataset_params["num_workers"]
        print("using " + str(worker_num))
    if "pin_memory" in dataset_params:
        pin_memory = dataset_params["pin_memory"]
        print("Pin memory set to true")
    if "early_stopping" in forecast_model.params:
        es = EarlyStopper(forecast_model.params["early_stopping"]['patience'])
    opt = pytorch_opt_dict[training_params["optimizer"]](
        forecast_model.model.parameters(), **training_params["optim_params"])
    criterion_init_params = {}
    if "criterion_params" in training_params:
        criterion_init_params = training_params["criterion_params"]
    criterion = pytorch_criterion_dict[training_params["criterion"]](**criterion_init_params)
    if "probabilistic" in forecast_model.params["model_params"] or "probabilistic" in forecast_model.params:
        probabilistic = True
    else:
        probabilistic = False
    max_epochs = training_params["epochs"]
    save_epoch = training_params["save_epoch"]
    if issubclass(type(forecast_model.training), Dataset):
        # this means we'll use PyTorch's DataLoader to load the data into batches in each epoch
        data_loader = DataLoader(forecast_model.training, batch_size=training_params["batch_size"], shuffle=True,
                                 sampler=None, batch_sampler=None, num_workers=worker_num, collate_fn=None,
                                 pin_memory=pin_memory, drop_last=False, timeout=0, worker_init_fn=None)
        if dataset_params["t_range_valid"] is not None:
            validation_data_loader = DataLoader(forecast_model.validation, batch_size=training_params["batch_size"],
                                                shuffle=False, sampler=None, batch_sampler=None, num_workers=worker_num,
                                                collate_fn=None, pin_memory=pin_memory, drop_last=False, timeout=0,
                                                worker_init_fn=None)
    else:
        # use Kuai's method in his WRR paper to iterate
        data_loader = forecast_model.training
        if dataset_params["t_range_valid"] is not None:
            validation_data_loader = forecast_model.validation
        test_data_loader = forecast_model.test_data
        # batch_size * rho must be smaller than ngrid * nt, if not, the value logged will be negative that is wrong
        batch_size = dataset_params["batch_size"]
        rho = dataset_params["forecast_history"]
        ngrid = data_loader.x.shape[0]
        nt = data_loader.x.shape[1]
        while batch_size * rho >= ngrid * nt:
            # try to use a smaller batch_size to make the model runnable
            batch_size = int(batch_size / 10)
        if batch_size < 1:
            batch_size = 1
        assert batch_size < ngrid
        n_iter_ep = int(np.ceil(np.log(0.01) / np.log(1 - batch_size * rho / ngrid / nt)))
    session_params = []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        if isinstance(data_loader, DataLoader):
            total_loss = torch_single_train(forecast_model.model, opt, criterion, data_loader, takes_target,
                                            multi_targets=num_targets, device=forecast_model.device,
                                            probablistic=probabilistic, forward_params=forward_params.copy())
        else:
            total_loss = kuai_single_train(forecast_model.model, opt, criterion, data_loader, takes_target,
                                           multi_targets=num_targets,
                                           forward_params={"n_iter_ep": n_iter_ep, "epoch": epoch, "ngrid": ngrid,
                                                           "nt": nt, "batch_size": batch_size, "rho": rho,
                                                           "device": forecast_model.device})
        log_str = 'Epoch {} Loss {:.3f} time {:.2f}'.format(epoch, total_loss, time.time() - t0)
        print(log_str)
        use_decoder = False
        if "use_decoder" in forecast_model.params:
            use_decoder = True
        if dataset_params["t_range_valid"] is not None:
            if isinstance(data_loader, validation_data_loader):
                valid = compute_validation(validation_data_loader, forecast_model.model, epoch,
                                           forecast_model.params["dataset_params"]["forecast_length"],
                                           forecast_model.crit,
                                           forecast_model.device, multi_targets=num_targets,
                                           decoder_structure=use_decoder,
                                           probabilistic=probabilistic)
            if valid == 0.0:
                raise ValueError("Error validation loss is zero there is a problem with the validator.")
            epoch_params = {"epoch": epoch, "train_loss": str(total_loss), "validation_loss": str(valid)}
            session_params.append(epoch_params)
            if es:
                if not es.check_loss(forecast_model.model, valid):
                    print("Stopping model now")
                    forecast_model.model.load_state_dict(torch.load("checkpoint.pth"))
                    break
        else:
            epoch_params = {"epoch": epoch, "train_loss": str(total_loss)}
            session_params.append(epoch_params)
        if save_epoch > 0 and epoch % save_epoch == 0:
            # save model
            model_file = os.path.join(model_filepath, 'model_Ep' + str(epoch) + '.pth')
            torch.save(forecast_model.model.state_dict(), model_file)
    forecast_model.params["run"] = session_params
    forecast_model.save_model(model_filepath, max_epochs)


def handle_scaling(validation_dataset, src, output: torch.Tensor, labels, probabilistic, m, output_std):
    # To-do move to class function
    output_dist = None
    if probabilistic:
        unscaled_out = validation_dataset.inverse_scale(output)
        try:
            output_std = numpy_to_tvar(output_std)
        except Exception:
            pass
        output_dist = torch.distributions.Normal(unscaled_out, output_std)
    elif m > 1:
        output = validation_dataset.inverse_scale(output.cpu())
        labels = validation_dataset.inverse_scale(labels.cpu())
    elif len(output.shape) == 3:
        output = output.cpu().numpy().transpose(0, 2, 1)
        labels = labels.cpu().numpy().transpose(0, 2, 1)
        output = validation_dataset.inverse_scale(torch.from_numpy(output))
        labels = validation_dataset.inverse_scale(torch.from_numpy(labels))
        stuff = src.cpu().numpy().transpose(0, 2, 1)
        src = validation_dataset.inverse_scale(torch.from_numpy(stuff))
    else:
        output = validation_dataset.inverse_scale(output.cpu().transpose(1, 0))
        labels = validation_dataset.inverse_scale(labels.cpu().transpose(1, 0))
        src = validation_dataset.inverse_scale(src.cpu().transpose(1, 0))
    return src, output, labels, output_dist


def compute_loss(labels, output, src, criterion, validation_dataset, probabilistic=None, output_std=None, m=1):
    """Function for computing the loss

    :param labels: The real values for the target. Shape can be variable but should follow (batch_size, time)
    :type labels: torch.Tensor
    :param output: The output of the model
    :type output: torch.Tensor
    :param src: The source values (only really needed for the MASELoss function)
    :type src: torch.Tensor
    :param criterion: [description]
    :type criterion: [type]
    :param validation_dataset: Only passed when unscaling of data is needed.
    :type validation_dataset: torch.utils.data.dataset
    :param probabilistic: Whether the model is a probabalistic returns a distribution, defaults to None
    :type probabilistic: [type], optional
    :param output_std: The standard distribution, defaults to None
    :type output_std: [type], optional
    :param m: [description], defaults to 1
    :type m: int, optional
    :return: Returns the computed loss
    :rtype: float
"""
    if isinstance(criterion, GaussianLoss):
        if len(output[0].shape) > 2:
            g_loss = GaussianLoss(output[0][:, :, 0], output[1][:, :, 0])
        else:
            g_loss = GaussianLoss(output[0][:, 0], output[1][:, 0])
        loss = g_loss(labels)
        return loss
    if not probabilistic and isinstance(output, torch.Tensor):
        if len(labels.shape) != len(output.shape):
            if len(labels.shape) > 1:
                if labels.shape[1] == output.shape[1]:
                    labels = labels.unsqueeze(2)
                else:
                    labels = labels.unsqueeze(0)
    if probabilistic:
        if type(output_std) != torch.Tensor:
            print("Converted tensor")
            output_std = torch.from_numpy(output_std)
        if type(output) != torch.Tensor:
            output = torch.from_numpy(output)
        output_dist = torch.distributions.Normal(output, output_std)
    if validation_dataset:
        src, output, labels, output_dist = handle_scaling(validation_dataset, src, output, labels,
                                                          probabilistic, m, output_std)
    if probabilistic:
        loss = -output_dist.log_prob(labels.float()).sum()  # FIX THIS?
    elif isinstance(criterion, MASELoss):
        assert len(labels.shape) == len(output.shape)
        loss = criterion(labels.float(), output, src, m)
    else:
        assert len(labels.shape) == len(output.shape)
        assert labels.shape[0] == output.shape[0]
        loss = criterion(output, labels.float())
    return loss


def kuai_single_train(model, opt, criterion, data_loader: HydroDlTsDataModel, takes_target, multi_targets,
                      forward_params):
    """model: a DL model"""
    n_iter_ep = forward_params["n_iter_ep"]
    iEpoch = forward_params["epoch"]
    ngrid = forward_params["ngrid"]
    nt = forward_params["nt"]
    batch_size = forward_params["batch_size"]
    rho = forward_params["rho"]
    device = forward_params["device"]
    loss_ep = 0
    for iIter in range(0, n_iter_ep):
        # training iterations
        i_grid, i_t = random_index(ngrid, nt, [batch_size, rho])
        if type(model) in sequence_first_model_lst:
            batch_first = False
        else:
            batch_first = True
        one_batch = data_loader.get_item(i_grid, i_t, rho, batch_first=batch_first)
        # Convert to CPU/GPU/TPU
        xy = [data_tmp.to(device) for data_tmp in one_batch]
        y_train = xy[-1]
        y_p = model(*xy[0:-1])
        if type(y_p) is tuple:
            others = y_p[1:]
            # Convention: y_p must be the first output of model
            y_p = y_p[0]
        loss = criterion(y_p, y_train)
        if torch.isnan(loss) or loss == float('inf'):
            raise ValueError("Error infinite or NaN loss detected. Try normalizing data or performing interpolation")
        loss.backward()
        opt.step()
        model.zero_grad()
        loss_ep = loss_ep + loss.item()
    # print loss
    loss_ep = loss_ep / n_iter_ep
    return loss_ep


def torch_single_train(model,
                       opt: optim.Optimizer,
                       criterion: Type[torch.nn.modules.loss._Loss],
                       data_loader: DataLoader,
                       takes_target: bool,
                       multi_targets=1, device=None, probablistic=None,
                       forward_params: Dict = {}) -> float:
    print('running torch_single_train')
    i = 0
    output_std = None
    running_loss = 0.0
    for i_batch, (src, trg) in enumerate(data_loader):
        # Convert to CPU/GPU/TPU
        src = src.to(device)
        trg = trg.to(device)
        if takes_target:
            forward_params["t"] = trg
        output = model(src, **forward_params)
        if probablistic:
            output1 = output
            output = output.mean
            output_std = output1.stddev
        loss = compute_loss(trg, output, src, criterion, None, probablistic, output_std, m=multi_targets)
        if loss > 100:
            print("Warning: high loss detected")
        loss.backward()
        opt.step()
        model.zero_grad()
        if torch.isnan(loss) or loss == float('inf'):
            raise ValueError("Error infinite or NaN loss detected. Try normalizing data or performing interpolation")
        running_loss += loss.item()
        i += 1
    print("The running loss is: ")
    print(running_loss)
    print("The number of items in train is: " + str(i))
    total_loss = running_loss / float(i)
    return total_loss


def compute_validation(validation_loader: DataLoader,
                       model,
                       epoch: int,
                       sequence_size: int,
                       criterion: Type[torch.nn.modules.loss._Loss],
                       device: torch.device,
                       decoder_structure=False,
                       use_wandb: bool = False,
                       multi_targets=1,
                       val_or_test="validation_loss",
                       probabilistic=False) -> float:
    """Function to compute the validation loss metrics

    :param validation_loader: The data-loader of either validation or test-data
    :type validation_loader: DataLoader
    :param model: model
    :type model: [type]
    :param epoch: [description]
    :type epoch: int
    :param sequence_size: [description]
    :type sequence_size: int
    :param criterion: [description]
    :type criterion: Type[torch.nn.modules.loss._Loss]
    :param device: [description]
    :type device: torch.device
    :param decoder_structure: [description], defaults to False
    :type decoder_structure: bool, optional
    :param meta_data_model: [description], defaults to None
    :type meta_data_model: [type], optional
    :param use_wandb: [description], defaults to False
    :type use_wandb: bool, optional
    :param meta_model: [description], defaults to None
    :type meta_model: [type], optional
    :param multi_targets: [description], defaults to 1
    :type multi_targets: int, optional
    :param val_or_test: [description], defaults to "validation_loss"
    :type val_or_test: str, optional
    :param probabilistic: [description], defaults to False
    :type probabilistic: bool, optional
    :return: The loss of the first metirc in the list.
    :rtype: float
    """
    print('Computing validation loss')
    unscaled_crit = dict.fromkeys(criterion, 0)
    scaled_crit = dict.fromkeys(criterion, 0)
    model.eval()
    output_std = None
    scaler = None
    if validation_loader.dataset.no_scale:
        scaler = validation_loader.dataset
    with torch.no_grad():
        i = 0
        loss_unscaled_full = 0.0
        for src, targ in validation_loader:
            src = src.to(device)
            targ = targ.to(device)
            i += 1
            if decoder_structure:
                if type(model).__name__ == "SimpleTransformer":
                    targ_clone = targ.detach().clone()
                    output = greedy_decode(model, src, targ.shape[1], targ_clone, device=device)[:, :, 0]
                else:
                    if probabilistic:
                        output, output_std = simple_decode(model, src, targ.shape[1], targ, 1,
                                                           multi_targets=multi_targets, probabilistic=probabilistic,
                                                           scaler=scaler)
                        output, output_std = output[:, :, 0], output_std[0]
                        output_dist = torch.distributions.Normal(output, output_std)
                    else:
                        output = simple_decode(model=model, src=src, max_seq_len=targ.shape[1], real_target=targ,
                                               output_len=sequence_size, multi_targets=multi_targets,
                                               probabilistic=probabilistic, scaler=scaler)
            else:
                if probabilistic:
                    output_dist = model(src.float())
                    output = output_dist.mean.detach().numpy()
                    output_std = output_dist.stddev.detach().numpy()
                else:
                    output = model(src.float())
            if multi_targets == 1:
                labels = targ[:, :, 0]
            elif multi_targets > 1:
                labels = targ[:, :, 0:multi_targets]
            validation_dataset = validation_loader.dataset
            for crit in criterion:
                if validation_dataset.scale:
                    # Should this also do loss.item() stuff?
                    if len(src.shape) == 2:
                        src = src.unsqueeze(0)
                    src1 = src[:, :, 0:multi_targets]
                    loss_unscaled_full = compute_loss(labels, output, src1, crit, validation_dataset, probabilistic,
                                                      output_std, m=multi_targets)
                    unscaled_crit[crit] += loss_unscaled_full.item() * len(labels.float())
                loss = compute_loss(labels, output, src, crit, False, probabilistic, output_std, m=multi_targets)
                scaled_crit[crit] += loss.item() * len(labels.float())
    model.train()
    return list(scaled_crit.values())[0]
