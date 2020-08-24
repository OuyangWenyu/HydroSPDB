"""main entry-point functions for physics-guided machine learning"""

import os

from hydroDL.model import crit, rnn, pgml_run, pgnn


def master_train_reservoir_water_balance(model_input, pre_trained_model_epoch=1):
    model_dict = model_input.data_model2.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_train = model_dict['train']

    # data
    x, y, c = model_input.load_data(model_dict)
    var_lst = model_input.data_model2.data_source.all_configs.get("attr_chosen")
    attr_index = var_lst.index("STOR_NOR_2009")
    initial_states = c[attr_index]
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]
    opt_model['nx'] = nx
    opt_model['ny'] = ny
    # loss
    loss_fun = crit.RmseLoss()

    # model
    out = os.path.join(model_dict['dir']['Out'], "model")
    if not os.path.isdir(out):
        os.mkdir(out)
    model = pgnn.CudnnWaterBalanceNN(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])

    # train model
    pgml_run.model_train_water_balance_nn(model, x, y, c, loss_fun, n_epoch=opt_train['nEpoch'],
                                          mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'],
                                          save_folder=out,
                                          pre_trained_model_epoch=pre_trained_model_epoch,
                                          initial_states=initial_states)
