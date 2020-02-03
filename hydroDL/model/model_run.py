"""模型调用的核心代码"""
import numpy as np
import torch
import time
import os
import pandas as pd

from . import rnn
from torch.utils.tensorboard import SummaryWriter


def model_test_for_lstm_without_1stlinear(model, x, c, *, file_path, batch_size=None):
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
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'a') as f:
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
            y_p = model(x_test)
            y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)

            # save output，目前只有一个变量径流，没有多个文件，所以直接取数据即可，因为DataFrame只能作用到二维变量，所以必须用y_out[:, :, 0]
            pd.DataFrame(y_out[:, :, 0]).to_csv(f, header=False, index=False)

            model.zero_grad()
            torch.cuda.empty_cache()

        f.close()
    y_out = torch.from_numpy(y_out)
    return y_out


def model_train_for_lstm_without_1stlinear(model, x, y, c, loss_fun, *, n_epoch=500, mini_batch=[100, 30],
                                           save_epoch=100, save_folder=None, mode='seq2seq'):
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
        run_file = os.path.join(save_folder, str(n_epoch) + 'epoch_run.csv')
        rf = open(run_file, 'a+')
    for iEpoch in range(1, n_epoch + 1):
        loss_ep = 0
        t0 = time.time()
        for iIter in range(0, n_iter_ep):
            # training iterations
            i_grid, i_t = random_index(ngrid, nt, [batch_size, rho])
            x_train = select_subset(x, i_grid, i_t, rho, c=c)
            y_train = select_subset(y, i_grid, i_t, rho)
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
                model_file = os.path.join(save_folder,
                                          'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, model_file)
    if save_folder is not None:
        rf.close()
    return model


def test_dataloader(net, testloader):
    test_obs = []
    test_preds = []
    if torch.cuda.is_available():
        net = net.cuda()
    with torch.no_grad():
        for data in testloader:
            xs, ys = data
            if torch.cuda.is_available():
                xs = xs.cuda()
                ys = ys.cuda()
            output = net(xs)
            test_obs.append(ys.cpu().numpy())
            test_preds.append(output.cpu().numpy())
    return test_preds, test_obs


def train_dataloader(net, trainloader, criterion, n_epoch, out_folder, save_model_folder, save_epoch):
    """train a model using data from dataloader"""
    print("Start Training...")
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        net = net.cuda()
    optimizer = torch.optim.Adadelta(net.parameters())
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(os.path.join(out_folder, 'runs', 'experiment_1'))
    # print structure of net
    dataiter = iter(trainloader)
    inputs4graph, outputs4graph = dataiter.next()
    if torch.cuda.is_available():
        writer.add_graph(net, inputs4graph.cuda())

    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        steps_num = 0
        for step, (batch_xs, batch_ys) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_xs = batch_xs.cuda()
                batch_ys = batch_ys.cuda()
            # forward + backward + optimize
            outputs = net(batch_xs)
            loss = criterion(outputs, batch_ys)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps_num = steps_num + 1
            print('Epoch: ', epoch, '| Step: ', step, '| loss_avg: ', running_loss / steps_num)

        # log the running loss
        writer.add_scalar('training loss', running_loss / steps_num, epoch * len(trainloader) + steps_num)
        # save model
        if epoch % save_epoch == save_epoch - 1:
            # save model, epoch count from 0, I wanna saved model counted from 1
            model_file = os.path.join(save_model_folder, 'model_Ep' + str(epoch + 1) + '.pt')
            torch.save(net, model_file)
    writer.close()
    print('Finished Training')


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
                mode='seq2seq'):
    batch_size, rho = mini_batch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    # batch_size * rho must be bigger than ngrid * nt, if not, the value logged will be negative  that is wrong
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
            if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel]:
                y_p = model(x_test)
            if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel]:
                y_p = model(x_test, z_test)
            if type(model) in [rnn.LstmCnnForcast]:
                y_p = model(x_test, z_test)
            y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)

            # save output，目前只有一个变量径流，没有多个文件，所以直接取数据即可，因为DataFrame只能作用到二维变量，所以必须用y_out[:, :, 0]
            pd.DataFrame(y_out[:, :, 0]).to_csv(f, header=False, index=False)

            model.zero_grad()
            torch.cuda.empty_cache()

        f.close()
    # TODO: y_out is not all output
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


def random_index(ngrid, nt, dim_subset):
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, [batch_size])
    i_t = np.random.randint(0, nt - rho, [batch_size])
    return i_grid, i_t


def select_subset(x, i_grid, i_t, rho, *, c=None, tuple_out=False):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(i_grid):  # hack
        i_grid = np.arange(0, len(i_grid))  # hack
        if nt <= rho:
            i_t.fill(0)
    if i_t is not None:
        batch_size = i_grid.shape[0]
        x_tensor = torch.zeros([rho, batch_size, nx], requires_grad=False)
        for k in range(batch_size):
            temp = x[i_grid[k]:i_grid[k] + 1, np.arange(i_t[k], i_t[k] + rho), :]
            x_tensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if len(x.shape) == 2:
            x_tensor = torch.from_numpy(x[i_grid, :]).float()
        else:
            x_tensor = torch.from_numpy(np.swapaxes(x[i_grid, :, :], 1, 0)).float()
            rho = x_tensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho, axis=1)
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


def model_train_inv(model, xqch, xct, qt, lossFun, *, n_epoch=500, mini_batch=[100, 30], save_epoch=100,
                    save_folder=None):
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
    for iEpoch in range(1, n_epoch + 1):
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
    if nt != nt_t:
        print("check")
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

    # save output，目前只有一个变量径流，没有多个文件，所以直接取数据即可，因为DataFrame只能作用到二维变量，所以必须用y_out[:, :, 0]
    model.zero_grad()
    torch.cuda.empty_cache()

    return y_out_list, param_list
