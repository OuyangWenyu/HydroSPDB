"""main entry-point functions for physics-guided machine learning"""

import os
import numpy as np
from hydroDL.model import crit, pgml_run, pgnn


def master_train_reservoir_water_balance(model_input, pre_trained_model_epoch=1):
    model_dict = model_input.data_model2.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_train = model_dict['train']

    # data
    forcing, outflow, c = model_input.data_model2.load_data(model_dict)
    var_lst = model_input.data_model2.data_source.all_configs.get("attr_chosen")
    attr_index = var_lst.index("STOR_NOR_2009")
    initial_states = c[:, attr_index]
    # the time length of a period in one iteration
    delta_t = np.array([10])
    # inflow is here
    inflow = model_input.natural_flow.reshape(model_input.natural_flow.shape[0], model_input.natural_flow.shape[1], 1)
    nx = inflow.shape[-1]
    ny = outflow.shape[-1]
    opt_model['nx'] = nx + 1
    opt_model['ny'] = ny
    # loss
    loss_fun = crit.RmseLoss()

    # model
    out = os.path.join(model_dict['dir']['Out'], "model")
    if not os.path.isdir(out):
        os.mkdir(out)
    model = pgnn.CudnnWaterBalanceNN(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'],
                                     iter_num=opt_train['miniBatch'][1], delta_t=delta_t)

    # train model
    pgml_run.model_train_water_balance_nn(model, inflow, outflow, c, initial_states, loss_fun,
                                          n_epoch=opt_train['nEpoch'],
                                          mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'],
                                          save_folder=out,
                                          pre_trained_model_epoch=pre_trained_model_epoch)


def generate_fake_outflow(model_input, init_stor_time_idx):
    print("a outflow generator with reservoir operation rules")
    model_dict = model_input.data_model2.data_source.data_config.model_dict
    # data
    forcing, y, c = model_input.data_model2.load_data(model_dict)
    var_lst = model_input.data_model2.data_source.all_configs.get("attr_chosen")
    attr_index = var_lst.index("STOR_NOR_2009")
    initial_states = c[:, attr_index]
    inflows = model_input.natural_flow
    capacity = model_input.data_model2.data_source.read_attr([""])
    targets = sim_demand()
    outflows, storages = sim_res(inflows, initial_states, targets, capacity)
    return outflows


def sim_demand():
    print("simulate the daily water demand")
    return []


def sim_res(inflow, storage0, target, capacity, policy=None):
    """simulate the reservoir operation, the default policy is SOP(standard operation policy),
     refer to:https://github.com/swd-turner/reservoir/blob/master/R/simres.R"""
    print("simulate reservoir operation with SOP")
    storage = np.zeros(len(inflow) + 1)
    storage[0] = storage0
    evapor = np.zeros(len(inflow))
    spill = np.zeros(len(inflow))
    release = np.zeros(len(inflow))
    if policy is None:
        for t in range(len(inflow)):
            evapor[t] = cal_evap()
            if storage[t] - target[t] + inflow[t] - evapor[t] > capacity:
                storage[t + 1] = capacity
                spill[t] = storage[t] - target[t] + inflow[t] - capacity - evapor[t]
                release[t] = target[t]
            else:
                if storage[t] - target[t] + inflow[t] - evapor[t] < 0:
                    storage[t + 1] = 0
                    release[t] = max(0, storage[t] + inflow[t] - evapor[t])
                else:
                    storage[t + 1] = storage[t] + inflow[t] - target[t] - evapor[t]
                    release[t] = target[t]
    else:
        print("STILL developing")
    outflow = spill + release
    return outflow, storage


def cal_evap():
    # TODO: calculate the ET for reservoirs in a basin
    return 0
