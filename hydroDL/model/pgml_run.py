import os
import time

import torch

from utils import my_timer
import numpy as np

from utils.hydro_math import random_index, select_subset


@my_timer
def model_train_water_balance_nn(model,
                                 x,
                                 y,
                                 c,
                                 initial_states,
                                 loss_fun,
                                 *,
                                 n_epoch=500,
                                 mini_batch=[100, 30],
                                 save_epoch=100,
                                 save_folder=None,
                                 pre_trained_model_epoch=1):
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
            # TODO: i_t should be the period with normal storage
            i_grid, i_t = random_index(ngrid, nt, [batch_size, rho])
            x_train = select_subset(x, i_grid, i_t, rho)
            initial_state = torch.from_numpy(initial_states[i_grid]).float()
            if torch.cuda.is_available():
                initial_state = initial_state.cuda()
            y_train = select_subset(y, i_grid, i_t, rho)
            gens, y_p = model(x_train, initial_state)

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
