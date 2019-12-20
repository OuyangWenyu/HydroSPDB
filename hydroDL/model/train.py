import numpy as np
import torch
import time
import os
import hydroDL
# import * 的时候就体现出__init__.py文件的作用了
from hydroDL.model import *
import pandas as pd


def train_model(model,
                x,
                y,
                c,
                loss_fun,
                *,
                n_epoch=500,
                mini_batch=[100, 30],
                save_epoch=100,
                save_folder=None,
                mode='seq2seq'):
    batch_size, rho = mini_batch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    # batch_size * rho must be bigger than ngrid * nt, if not, the value logged will be negative  that is wrong
    n_iter_ep = int(
        np.ceil(np.log(0.01) / np.log(1 - batch_size * rho / ngrid / nt)))
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            n_iter_ep = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batch_size *
                                          (rho - model.ct) / ngrid / nt)))

    if torch.cuda.is_available():
        loss_fun = loss_fun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if save_folder is not None:
        run_file = os.path.join(save_folder, 'run.csv')
        rf = open(run_file, 'a+')
    for iEpoch in range(1, n_epoch + 1):
        loss_ep = 0
        t0 = time.time()
        for iIter in range(0, n_iter_ep):
            # training iterations
            if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel]:
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
        log_str = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, loss_ep,
            time.time() - t0)
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


def save_model(out_folder, model, epoch, model_name='model'):
    model_file = os.path.join(out_folder, model_name + '_Ep' + str(epoch) + '.pt')
    torch.save(model, model_file)


def load_model(out_folder, epoch, model_name='model'):
    model_file = os.path.join(out_folder, model_name + '_Ep' + str(epoch) + '.pt')
    model = torch.load(model_file)
    return model


def test_model(model, x, c, *, batch_size=None, file_path_lst=None):
    if type(x) is tuple or type(x) is list:
        x, z = x
    else:
        z = None
    ngrid, nt, nx = x.shape
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
    if file_path_lst is None:
        file_path_lst = ['out' + str(x) for x in range(ny)]
    f_lst = list()
    for filePath in file_path_lst:
        if os.path.exists(filePath):
            os.remove(filePath)
        f = open(filePath, 'a')
        f_lst.append(f)

    # forward for each batch
    for i in range(0, len(i_s)):
        print('batch {}'.format(i))
        x_temp = x[i_s[i]:i_e[i], :, :]
        c_temp = np.repeat(
            np.reshape(c[i_s[i]:i_e[i], :], [i_e[i] - i_s[i], 1, nc]), nt, axis=1)
        x_test = torch.from_numpy(
            np.swapaxes(np.concatenate([x_temp, c_temp], 2), 1, 0)).float()
        if torch.cuda.is_available():
            x_test = x_test.cuda()
        if z is not None:
            z_temp = z[i_s[i]:i_e[i], :, :]
            z_test = torch.from_numpy(np.swapaxes(z_temp, 1, 0)).float()
            if torch.cuda.is_available():
                z_test = z_test.cuda()
        if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel]:
            y_p = model(x_test)
        if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel]:
            y_p = model(x_test, z_test)
        if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            y_p = model(x_test, z_test)
        y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)

        # save output
        for k in range(ny):
            f = f_lst[k]
            pd.DataFrame(y_out[:, :, k]).to_csv(f, header=False, index=False)

        model.zero_grad()
        torch.cuda.empty_cache()

    for f in f_lst:
        f.close()

    y_out = torch.from_numpy(y_out)
    return y_out


def test_model_cnn_cond(model, x, y, *, batch_size=None):
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


def random_index(ngrid, nt, dim_subset):
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, [batch_size])
    if nt <= rho:
        i_t = np.random.randint(0, 1, [batch_size])
    else:
        i_t = np.random.randint(0, nt - rho, [batch_size])
    return i_grid, i_t


def select_subset(x, i_grid, i_t, rho, *, c=None, tuple_out=False):
    nx = x.shape[-1]
    if x.shape[0] == len(i_grid):  # hack
        i_grid = np.arange(0, len(i_grid))  # hack
        i_t.fill(0)
    if i_t is not None:
        batch_size = i_grid.shape[0]
        x_tensor = torch.zeros([rho, batch_size, nx], requires_grad=False)
        for k in range(batch_size):
            temp = x[i_grid[k]:i_grid[k] + 1, np.arange(i_t[k], i_t[k] + rho), :]
            x_tensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        x_tensor = torch.from_numpy(np.swapaxes(x[i_grid, :, :], 1, 0)).float()
        rho = x_tensor.shape[1]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho, axis=1)
        c_tensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()
        if tuple_out:
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                c_tensor = c_tensor.cuda()
            out = (x_tensor, c_tensor)
        else:
            out = torch.cat((x_tensor, c_tensor), 2)
    else:
        out = x_tensor
    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out
