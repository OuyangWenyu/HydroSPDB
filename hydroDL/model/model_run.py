"""call the models"""
import numpy as np
import torch
import time
import os
import pandas as pd

from utils import my_timer
from utils.hydro_math import random_index, select_subset, select_subset_batch_first, select_subset_seq
from . import rnn
from torch.utils.tensorboard import SummaryWriter

from .early_stopping import EarlyStopping


def train_valid_dataloader(net, trainloader, validloader, criterion, n_epoch, out_folder, save_epoch, seq_first=True):
    """train a model using data from dataloader"""
    print("Start Training...")
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=save_epoch, verbose=True)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        net = net.cuda()
    optimizer = torch.optim.Adadelta(net.parameters())
    for epoch in range(1, n_epoch + 1):  # loop over the dataset multiple times
        ###################
        # train the model #
        ###################
        net.train()  # prep model for training
        t0 = time.time()
        for step, (batch_xs, batch_ys) in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            optimizer.zero_grad()
            if seq_first:
                batch_xs = batch_xs.transpose(0, 1)
                batch_ys = batch_ys.transpose(0, 1)
            if torch.cuda.is_available():
                batch_xs = batch_xs.cuda()
                batch_ys = batch_ys.cuda()
            # forward + backward + optimize
            outputs = net(batch_xs)
            loss = criterion(outputs, batch_ys)
            loss.backward()
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            # print('Epoch: ', epoch, '| Step: ', step, '| loss_avg: ', running_loss / steps_num)

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        for data, target in validloader:
            if seq_first:
                data = data.transpose(0, 1)
                target = target.transpose(0, 1)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(n_epoch))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        log_str = 'time {:.2f}'.format(time.time() - t0)
        print(print_msg, log_str)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, net, out_folder)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # load the last checkpoint with the best model
    net.load_state_dict(torch.load(os.path.join(out_folder, 'checkpoint.pt')))
    print('Finished Training')
    return net, avg_train_losses, avg_valid_losses


def test_dataloader(net, testloader, batch_first=False):
    # batch_first means the model not the input, the inputs are always batch_first here.
    test_obs = []
    test_preds = []
    if torch.cuda.is_available():
        net = net.cuda()
    with torch.no_grad():
        for data in testloader:
            xs, ys = data
            if not batch_first:
                xs = xs.transpose(0, 1)
                ys = ys.transpose(0, 1)
            if torch.cuda.is_available():
                xs = xs.cuda()
                ys = ys.cuda()
            output = net(xs)
            if not batch_first:
                test_obs.append(np.swapaxes(ys.cpu().numpy(), 1, 0))
                test_preds.append(np.swapaxes(output.cpu().numpy(), 1, 0))
            else:
                test_obs.append(ys.cpu().numpy())
                test_preds.append(output.cpu().numpy())
    return test_preds, test_obs


def train_dataloader(net, trainloader, criterion, n_epoch, out_folder, save_model_folder, save_epoch,
                     batch_first=False):
    """train a model using data from dataloader"""
    print("Start Training...")
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        net = net.cuda()
    optimizer = torch.optim.Adadelta(net.parameters())
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(os.path.join(out_folder, 'runs', 'experiment_1'))
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        steps_num = 0
        t0 = time.time()
        for step, (batch_xs, batch_ys) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_xs = batch_xs.cuda()
                batch_ys = batch_ys.cuda()
            # forward + backward + optimize
            if not batch_first:
                batch_xs = batch_xs.transpose(0, 1)
                batch_ys = batch_ys.transpose(0, 1)
            outputs = net(batch_xs)
            loss = criterion(outputs, batch_ys)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps_num = steps_num + 1
            print('Epoch: ', epoch, '| Step: ', step, '| loss_avg: ', running_loss / steps_num)
        loss_ep = running_loss / steps_num
        log_str = 'Epoch {} Loss {:.3f} time {:.2f}'.format(epoch, loss_ep, time.time() - t0)
        print("\n", log_str, "\n")
        # log the running loss
        writer.add_scalar('training loss', running_loss / steps_num, epoch * len(trainloader) + steps_num)
        # save model
        if epoch % save_epoch == save_epoch - 1:
            # save model, epoch count from 0, I wanna saved model counted from 1
            model_file = os.path.join(save_model_folder, 'model_Ep' + str(epoch + 1) + '.pt')
            torch.save(net, model_file)
    writer.close()
    print('Finished Training')


def model_train_valid(model, x, y, c, loss_fun, *, n_epoch=500, mini_batch=[100, 30], save_epoch=100, save_folder=None,
                      mode='seq2seq', gpu_num=1, valid_size=0.2):
    batch_size, rho = mini_batch
    patience = save_epoch
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    # batch_size * rho must be bigger than ngrid * nt, if not, the value logged will be negative  that is wrong
    n_iter_ep = int(np.ceil(np.log(0.01) / np.log(1 - batch_size * rho / ngrid / nt)))
    if torch.cuda.is_available():
        loss_fun = loss_fun.cuda()
        model = model.cuda()
    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for iEpoch in range(1, n_epoch + 1):
        # split time to train and valid
        num_train = nt - rho
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        assert len(train_idx) > batch_size
        assert len(valid_idx) > batch_size
        #  use i_grids to keep same grids in train and valid
        i_grids = []
        ###################
        # train the model #
        ###################
        t0 = time.time()
        model.train()  # prep model for training
        for iIter in range(0, n_iter_ep):
            i_t = np.array(train_idx)[np.random.randint(0, len(train_idx), [batch_size])]
            i_grid = np.random.randint(0, ngrid, [batch_size])
            i_grids.append(i_grid)

            x_train = select_subset(x, i_grid, i_t, rho, c=c)
            y_train = select_subset(y, i_grid, i_t, rho)
            y_p = model(x_train)
            loss = loss_fun(y_p, y_train)
            loss.backward()
            optim.step()
            model.zero_grad()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for iIter in range(0, n_iter_ep):
            i_grid = i_grids[iIter]
            i_t = np.array(valid_idx)[np.random.randint(0, len(valid_idx), [batch_size])]
            x_valid = select_subset(x, i_grid, i_t, rho, c=c)
            y_valid = select_subset(y, i_grid, i_t, rho)
            y_output = model(x_valid)
            # calculate the loss
            loss = loss_fun(y_output, y_valid)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epoch))

        print_msg = (f'[{iEpoch:>{epoch_len}}/{n_epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        log_str = 'time {:.2f}'.format(time.time() - t0)
        print(print_msg, log_str)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model, save_folder)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(os.path.join(save_folder, 'checkpoint.pt')))
    return model, avg_train_losses, avg_valid_losses


def model_test_valid(model, x, c, *, file_path, batch_size=None):
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    ny = model.ny
    if batch_size is None:
        batch_size = ngrid
    if torch.cuda.is_available():
        model = model.cuda()
    # y_p = torch.zeros([nt, ngrid, ny])
    i_s = np.arange(0, ngrid, batch_size)
    i_e = np.append(i_s[1:], ngrid)

    # deal with file name to save
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'a') as f:
        # forward for each batch
        for i in range(0, len(i_s)):
            print('batch {}'.format(i))
            x_temp = x[i_s[i]:i_e[i], :, :]
            if c is not None:
                c_temp = np.repeat(np.reshape(c[i_s[i]:i_e[i], :], [i_e[i] - i_s[i], 1, nc]), nt, axis=1)
                x_test = torch.from_numpy(np.swapaxes(np.concatenate([x_temp, c_temp], 2), 1, 0)).float()
            else:
                x_test = torch.from_numpy(np.swapaxes(x_temp, 1, 0)).float()
            if torch.cuda.is_available():
                x_test = x_test.cuda()

            y_p = model(x_test)
            y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
            # save output，now only for streamflow: y_out[:, :, 0]
            pd.DataFrame(y_out[:, :, 0]).to_csv(f, header=False, index=False)

            model.zero_grad()
            torch.cuda.empty_cache()

        f.close()
    # TODO: y_out is not all output
    y_out = torch.from_numpy(y_out)
    return y_out


@my_timer
def model_train(model,
                x,
                y,
                c,
                loss_fun,
                *,
                n_epoch=500,
                mini_batch=[100, 30],
                save_epoch=100,
                save_folder=None,
                pre_trained_model_epoch=1,
                mode='seq2seq',
                gpu_num=1):
    batch_size, rho = mini_batch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    # batch_size * rho must be smaller than ngrid * nt, if not, the value logged will be negative  that is wrong
    while batch_size * rho >= ngrid * nt:
        # try to use a smaller batch_size to make the model runnable
        batch_size = int(batch_size / 10)
    if batch_size < 1:
        batch_size = 1
    n_iter_ep = int(np.ceil(np.log(0.01) / np.log(1 - batch_size * rho / ngrid / nt)))
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            n_iter_ep = int(np.ceil(np.log(0.01) / np.log(1 - batch_size * (rho - model.ct) / ngrid / nt)))
    if torch.cuda.is_available():
        loss_fun = loss_fun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if save_folder is not None:
        run_file = os.path.join(save_folder, str(n_epoch) + 'epoch_run.csv')
        rf = open(run_file, 'a+')
    for iEpoch in range(pre_trained_model_epoch, n_epoch + 1):
        loss_ep = 0
        t0 = time.time()
        for iIter in range(0, n_iter_ep):
            # training iterations
            if type(model) in [rnn.CudnnLstmModel, rnn.CudnnLstmModelPretrain]:
                i_grid, i_t = random_index(ngrid, nt, [batch_size, rho])
                x_train = select_subset(x, i_grid, i_t, rho, c=c)
                y_train = select_subset(y, i_grid, i_t, rho)
                y_p = model(x_train)
            elif type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel]:
                i_grid, i_t = random_index(ngrid, nt, [batch_size, rho])
                x_train = select_subset(x, i_grid, i_t, rho, c=c)
                y_train = select_subset(y, i_grid, i_t, rho)
                z_train = select_subset(z, i_grid, i_t, rho)
                y_p = model(x_train, z_train)
            elif type(model) in [rnn.CudnnLstmModel_R2P]:
                i_grid, i_t = random_index(ngrid, nt, [batch_size, rho])
                x_train = select_subset(x, i_grid, i_t, rho, c=c, tuple_out=True)
                y_train = select_subset(y, i_grid, i_t, rho)
                y_p = model(x_train)

            # if type(model) in [hydroDL.model.rnn.LstmCnnCond]:
            #     i_grid, i_t = randomIndex(ngrid, nt, [batch_size, rho])
            #     x_train = selectSubset(x, i_grid, i_t, rho)
            #     y_train = selectSubset(y, i_grid, i_t, rho)
            #     z_train = selectSubset(z, i_grid, None, None)
            #     y_p = model(x_train, z_train)
            # if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            #     i_grid, i_t = randomIndex(ngrid, nt, [batch_size, rho])
            #     x_train = selectSubset(x, i_grid, i_t, rho)
            #     y_train = selectSubset(y, i_grid, i_t + model.ct, rho - model.ct)
            #     z_train = selectSubset(z, i_grid, i_t, rho)
            #     y_p = model(x_train, z_train)
            else:
                Exception('unknown model')
            loss = loss_fun(y_p, y_train)
            loss.backward()
            optim.step()
            model.zero_grad()
            loss_ep = loss_ep + loss.item()
        # print loss
        loss_ep = loss_ep / n_iter_ep
        log_str = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEpoch, loss_ep, time.time() - t0)
        print(log_str)
        # save model and loss
        if save_folder is not None:
            rf.write(log_str + '\n')
            if iEpoch % save_epoch == 0:
                # save model
                model_file = os.path.join(save_folder,
                                          'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, model_file)
    if save_folder is not None:
        rf.close()
    return model


def model_save(out_folder, model, epoch, model_name='model'):
    model_file = os.path.join(out_folder, model_name + '_Ep' + str(epoch) + '.pt')
    torch.save(model, model_file)


def model_load(out_folder, epoch, model_name='model'):
    model_file = os.path.join(out_folder, model_name + '_Ep' + str(epoch) + '.pt')
    model = torch.load(model_file)
    return model


def model_test(model, x, c, *, file_path, batch_size=None):
    if type(x) is tuple or type(x) is list:
        x, z = x
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    ny = model.ny
    if batch_size is None:
        batch_size = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct
    # y_p = torch.zeros([nt, ngrid, ny])
    i_s = np.arange(0, ngrid, batch_size)
    i_e = np.append(i_s[1:], ngrid)

    # deal with file name to save
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'a') as f:
        # forward for each batch
        for i in range(0, len(i_s)):
            print('batch {}'.format(i))
            x_temp = x[i_s[i]:i_e[i], :, :]
            if c is not None:
                c_temp = np.repeat(np.reshape(c[i_s[i]:i_e[i], :], [i_e[i] - i_s[i], 1, nc]), nt, axis=1)
                x_test = torch.from_numpy(np.swapaxes(np.concatenate([x_temp, c_temp], 2), 1, 0)).float()
            else:
                x_test = torch.from_numpy(np.swapaxes(x_temp, 1, 0)).float()
            if torch.cuda.is_available():
                x_test = x_test.cuda()
            if z is not None:
                z_temp = z[i_s[i]:i_e[i], :, :]
                z_test = torch.from_numpy(np.swapaxes(z_temp, 1, 0)).float()
                if torch.cuda.is_available():
                    z_test = z_test.cuda()
            if type(model) in [rnn.CudnnLstmModel, rnn.CudnnLstmModelPretrain]:
                y_p = model(x_test)
            if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel]:
                y_p = model(x_test, z_test)
            if type(model) in [rnn.LstmCnnForcast]:
                y_p = model(x_test, z_test)
            y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)

            pd.DataFrame(y_out[:, :, 0]).to_csv(f, header=False, index=False)

            model.zero_grad()
            torch.cuda.empty_cache()

        f.close()
    # y_out is not all output, so we use "pd.DataFrame(y_out[:, :, 0]).to_csv" to save result of each batch which will be used later
    y_out = torch.from_numpy(y_out)
    return y_out


def model_cnn_cond_test(model, x, y, *, batch_size=None):
    ngrid, nt, nx = x.shape
    ct = model.ct
    ny = model.ny
    if batch_size is None:
        batch_size = ngrid
    x_test = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
    # c_test = torch.from_numpy(np.swapaxes(y[:, 0:ct, :], 1, 0)).float()
    c_test = torch.zeros([ct, ngrid, y.shape[-1]], requires_grad=False)
    for k in range(ngrid):
        ctemp = y[k, 0:ct, 0]
        i0 = np.where(np.isnan(ctemp))[0]
        i1 = np.where(~np.isnan(ctemp))[0]
        if len(i1) > 0:
            ctemp[i0] = np.interp(i0, i1, ctemp[i1])
            c_test[:, k, 0] = torch.from_numpy(ctemp)

    if torch.cuda.is_available():
        x_test = x_test.cuda()
        c_test = c_test.cuda()
        model = model.cuda()

    model.train(mode=False)

    y_p = torch.zeros([nt - ct, ngrid, ny])
    i_s = np.arange(0, ngrid, batch_size)
    i_e = np.append(i_s[1:], ngrid)
    for i in range(0, len(i_s)):
        x_temp = x_test[:, i_s[i]:i_e[i], :]
        c_temp = c_test[:, i_s[i]:i_e[i], :]
        y_p[:, i_s[i]:i_e[i], :] = model(x_temp, c_temp)
    y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
    return y_out


def random_subset(x, y, dim_subset):
    ngrid, nt, nx = x.shape
    batch_size, rho = dim_subset
    x_tensor = torch.zeros([rho, batch_size, x.shape[-1]], requires_grad=False)
    y_tensor = torch.zeros([rho, batch_size, y.shape[-1]], requires_grad=False)
    i_grid = np.random.randint(0, ngrid, [batch_size])
    i_t = np.random.randint(0, nt - rho, [batch_size])
    for k in range(batch_size):
        temp = x[i_grid[k]:i_grid[k] + 1, np.arange(i_t[k], i_t[k] + rho), :]
        x_tensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[i_grid[k]:i_grid[k] + 1, np.arange(i_t[k], i_t[k] + rho), :]
        y_tensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    if torch.cuda.is_available():
        x_tensor = x_tensor.cuda()
        y_tensor = y_tensor.cuda()
    return x_tensor, y_tensor


def model_train_storage(model, qx, c, natflow, y, lossFun, *, seq_length_storage=100, n_epoch=500, mini_batch=[100, 30],
                        save_epoch=100, save_folder=None, pre_trained_model_epoch=1):
    batchSize, rho = mini_batch
    ngrid, nt, nx = qx.shape
    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / nt)))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if save_folder is not None:
        runFile = os.path.join(save_folder, str(n_epoch) + 'epoch_run.csv')
        rf = open(runFile, 'w+')
    for iEpoch in range(pre_trained_model_epoch, n_epoch + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            # training iterations
            iGrid, iT = random_index(ngrid, nt, [batchSize, rho])
            xTrain = select_subset(qx, iGrid, iT, rho, c=c)
            # iGrid and iT of xTrain and x_storage_train should be same
            x_storage_train = select_subset_seq(natflow, iGrid, iT, rho, c=c)
            # iTs of xtTrain and yTrain should be same
            yTrain = select_subset(y, iGrid, iT, rho)
            yP, Param_storage = model(x_storage_train, xTrain)  # will also send in the y for inversion generator
            loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
        # print loss
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEpoch, lossEp, time.time() - t0)
        print(logStr)
        # save model and loss
        if save_folder is not None:
            rf.write(logStr + '\n')
            if iEpoch % save_epoch == 0:
                # save model
                modelFile = os.path.join(save_folder, 'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)
    if save_folder is not None:
        rf.close()
    return model


def model_test_storage(model, qx, c, natflow, seq_len, batch_size):
    ngrid, nt, nx = qx.shape
    if c is not None:
        nc = c.shape[-1]
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    i_s = np.arange(0, ngrid, batch_size)
    i_e = np.append(i_s[1:], ngrid)

    y_out_list = []
    param_list = []
    for i in range(0, len(i_s)):
        print('batch {}'.format(i))
        xt_temp = qx[i_s[i]:i_e[i], :, :]
        # len of natflow and qx are different
        xh_2d_temp = natflow[i_s[i]:i_e[i], :]
        xh_temp = np.zeros([xt_temp.shape[0], xt_temp.shape[1], seq_len])

        for k in range(xh_temp.shape[1]):
            xh_temp[:, k, :] = xh_2d_temp[:, k:k + seq_len]

        # for j in range(xh_temp.shape[0]):
        #     xh_every_site = xh_2d_temp[j:j + 1, :]
        #     for k in range(xh_temp.shape[1]):
        #         xh_temp[j, k, :] = xh_every_site[:, k:k + seq_len]

        if c is not None:
            c_temp = np.repeat(np.reshape(c[i_s[i]:i_e[i], :], [i_e[i] - i_s[i], 1, nc]), nt, axis=1)
            xhTest = torch.from_numpy(np.swapaxes(np.concatenate([xh_temp, c_temp], 2), 1, 0)).float()
            xtTest = torch.from_numpy(np.swapaxes(np.concatenate([xt_temp, c_temp], 2), 1, 0)).float()
        else:
            xhTest = torch.from_numpy(np.swapaxes(xh_temp, 1, 0)).float()
            xtTest = torch.from_numpy(np.swapaxes(xt_temp, 1, 0)).float()
        if torch.cuda.is_available():
            xhTest = xhTest.cuda()
            xtTest = xtTest.cuda()

        y_p, param = model(xhTest, xtTest)
        y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
        param_out = param.detach().cpu().numpy().swapaxes(0, 1)

        y_out_list.append(y_out)
        param_list.append(param_out)

    model.zero_grad()
    torch.cuda.empty_cache()
    return y_out_list, param_list


def model_train_inv(model, xqch, xct, qt, lossFun, *, n_epoch=500, mini_batch=[100, 30], save_epoch=100,
                    save_folder=None, pre_trained_model_epoch=1):
    batchSize, rho = mini_batch
    ngrid, nt, nx = xqch.shape
    ngrid_t, nt_t, nx_t = xct.shape
    nt_temp = max(nt, nt_t)
    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / nt_temp)))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if save_folder is not None:
        runFile = os.path.join(save_folder, str(n_epoch) + 'epoch_run.csv')
        rf = open(runFile, 'w+')
    for iEpoch in range(pre_trained_model_epoch, n_epoch + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            # training iterations
            iGrid, iT = random_index(ngrid, nt, [batchSize, rho])
            xhTrain = select_subset(xqch, iGrid, iT, rho)
            iGrid_t, iT_t = random_index(ngrid_t, nt_t, [batchSize, rho])
            # iGrid should be same, iT can be different
            xtTrain = select_subset(xct, iGrid, iT_t, rho)
            # iTs of xtTrain and yTrain should be same
            yTrain = select_subset(qt, iGrid, iT_t, rho)
            yP, Param_Inv = model(xhTrain, xtTrain)  # will also send in the y for inversion generator
            loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
        # print loss
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEpoch, lossEp, time.time() - t0)
        print(logStr)
        # save model and loss
        if save_folder is not None:
            rf.write(logStr + '\n')
            if iEpoch % save_epoch == 0:
                # save model
                modelFile = os.path.join(save_folder, 'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)
    if save_folder is not None:
        rf.close()
    return model


def model_test_inv(model, xqch, xct, batch_size):
    ngrid, nt, nx = xqch.shape
    ngrid_t, nt_t, nx_t = xct.shape
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    i_s = np.arange(0, ngrid, batch_size)
    i_e = np.append(i_s[1:], ngrid)

    y_out_list = []
    param_list = []
    for i in range(0, len(i_s)):
        print('batch {}'.format(i))
        xh_temp = xqch[i_s[i]:i_e[i], :, :]
        xt_temp = xct[i_s[i]:i_e[i], :, :]
        # TODO: default shape1 of xh_temp is larger than xt_temp's
        len_max = xh_temp.shape[1]
        len_min = xt_temp.shape[1]
        time_batch_size = int(len_max / len_min) if len_max % len_min == 0 else int(len_max / len_min) + 1
        y_outs = []
        param_outs = []
        for j in range(time_batch_size):
            if j == time_batch_size - 1:
                xh_temp_j = xh_temp[:, j * len_min:, :]
                xt_temp_j = xt_temp
                # cut to same length
                if xh_temp_j.shape[1] != xt_temp_j.shape[1]:
                    xt_temp_j = xt_temp_j[:, :xh_temp_j.shape[1], :]
            else:
                xh_temp_j = xh_temp[:, j * len_min:(j + 1) * len_min, :]
                xt_temp_j = xt_temp
            xhTest = torch.from_numpy(np.swapaxes(xh_temp_j, 1, 0)).float()
            xtTest = torch.from_numpy(np.swapaxes(xt_temp_j, 1, 0)).float()
            if torch.cuda.is_available():
                xhTest = xhTest.cuda()
                xtTest = xtTest.cuda()

            y_p, param = model(xhTest, xtTest)
            y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
            param_out = param.detach().cpu().numpy().swapaxes(0, 1)
            y_outs.append(y_out)
            param_outs.append(param_out)

        # y_outs every items average
        def avg_3darray_list(arr_list):
            arr = arr_list[0]
            for i in range(1, len(arr_list) - 1):
                arr = arr + arr_list[i]
            arr = arr / (len(arr_list) - 1)
            arr_last = arr_list[len(arr_list) - 1]
            arr[:, :arr_last.shape[1], :] = (arr[:, :arr_last.shape[1], :] + arr_last) / 2
            return arr

        y_out_i = avg_3darray_list(y_outs)
        param_out_i = avg_3darray_list(param_outs)
        y_out_list.append(y_out_i)
        param_list.append(param_out_i)

    model.zero_grad()
    torch.cuda.empty_cache()

    return y_out_list, param_list


def model_test_inv_kernel(model, xqch, xct, batch_size):
    ngrid, nt, nx = xqch.shape
    ngrid_t, nt_t, nx_t = xct.shape
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    i_s = np.arange(0, ngrid, batch_size)
    i_e = np.append(i_s[1:], ngrid)

    y_out_list = []
    param_list = []
    for i in range(0, len(i_s)):
        print('batch {}'.format(i))
        xh_temp = xqch[i_s[i]:i_e[i], :, :]
        xt_temp = xct[i_s[i]:i_e[i], :, :]

        xhTest = torch.from_numpy(np.swapaxes(xh_temp, 1, 0)).float()
        xtTest = torch.from_numpy(np.swapaxes(xt_temp, 1, 0)).float()
        if torch.cuda.is_available():
            xhTest = xhTest.cuda()
            xtTest = xtTest.cuda()

        y_p, param = model(xhTest, xtTest)
        y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
        param_out = param.detach().cpu().numpy().swapaxes(0, 1)

        y_out_list.append(y_out)
        param_list.append(param_out)

    model.zero_grad()
    torch.cuda.empty_cache()
    return y_out_list, param_list


def model_train_batch1st_lstm(model,
                              x,
                              y,
                              c,
                              loss_fun,
                              *,
                              n_epoch=500,
                              mini_batch=[100, 30],
                              save_epoch=100,
                              save_folder=None,
                              mode='seq2seq',
                              gpu_num=1):
    batch_size, rho = mini_batch
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    # batch_size * rho must be bigger than ngrid * nt, if not, the value logged will be negative  that is wrong
    n_iter_ep = int(np.ceil(np.log(0.01) / np.log(1 - batch_size * rho / ngrid / nt)))
    if torch.cuda.is_available():
        loss_fun = loss_fun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if save_folder is not None:
        run_file = os.path.join(save_folder, str(n_epoch) + 'epoch_run.csv')
        rf = open(run_file, 'a+')
    for iEpoch in range(1, n_epoch + 1):
        loss_ep = 0
        t0 = time.time()
        for iIter in range(0, n_iter_ep):
            # training iterations
            i_grid, i_t = random_index(ngrid, nt, [batch_size, rho])
            x_train = select_subset_batch_first(x, i_grid, i_t, rho, c=c)
            y_train = select_subset_batch_first(y, i_grid, i_t, rho)
            y_p = model(x_train)
            loss = loss_fun(y_p, y_train)
            loss.backward()
            optim.step()
            model.zero_grad()
            loss_ep = loss_ep + loss.item()
        # print loss
        loss_ep = loss_ep / n_iter_ep
        log_str = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, loss_ep,
            time.time() - t0)
        print(log_str)
        # save model and loss
        if save_folder is not None:
            rf.write(log_str + '\n')
            if iEpoch % save_epoch == 0:
                # save model
                model_file = os.path.join(save_folder, 'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, model_file)
    if save_folder is not None:
        rf.close()
    return model


def model_test_batch1st_lstm(model, x, c, *, file_path, batch_size=None):
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    ny = model.ny
    if batch_size is None:
        batch_size = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    y_out_list = []
    i_s = np.arange(0, ngrid, batch_size)
    i_e = np.append(i_s[1:], ngrid)

    # deal with file name to save
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'a') as f:
        # forward for each batch
        for i in range(0, len(i_s)):
            print('batch {}'.format(i))
            x_temp = x[i_s[i]:i_e[i], :, :]
            if c is not None:
                c_temp = np.repeat(np.reshape(c[i_s[i]:i_e[i], :], [i_e[i] - i_s[i], 1, nc]), nt, axis=1)
                x_test = torch.from_numpy(np.concatenate([x_temp, c_temp], 2)).float()
            else:
                x_test = torch.from_numpy(x_temp).float()
            if torch.cuda.is_available():
                x_test = x_test.cuda()
            y_p = model(x_test)
            y_out = y_p.detach().cpu().numpy()
            y_out_list.append(y_out)
            pd.DataFrame(y_out[:, :, 0]).to_csv(f, header=False, index=False)

            model.zero_grad()
            torch.cuda.empty_cache()

        f.close()
    return y_out_list
