"""调用训练，测试，读取模型等函数的函数"""
import os
import numpy as np
import pandas as pd
import torch
from functools import reduce

from torch.utils.data import DataLoader

from data.data_config import name_pred
from data.data_input import _trans_norm, create_datasets, _basin_norm
from explore import stat
from hydroDL.model import *


def master_test_1by1(data_model):
    """:parameter
        data_model：测试使用的数据
        model_dict：测试时的模型配置
    """
    model_dict = data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    # generate file names and run model
    out = model_dict['dir']['Out']
    t_range = data_model.t_s_dict["t_final_range"]
    epoch = model_dict['train']["nEpoch"]
    file_path = name_pred(model_dict, out, t_range, epoch)
    print('output files:', file_path)

    model_file = os.path.join(out, 'checkpoint.pt')
    opt_model['nx'] = data_model.data_forcing.shape[-1]
    opt_model['ny'] = 1

    if opt_model['name'] == 'CudnnLstmModel':
        model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])
    elif opt_model['name'] == 'LstmCloseModel':
        model = rnn.LstmCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                   fillObs=True)
    elif opt_model['name'] == 'AnnModel':
        model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'])
    elif opt_model['name'] == 'AnnCloseModel':
        model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                  fillObs=True)
    model.load_state_dict(torch.load(model_file))
    testloader = create_datasets(data_model, train_mode=False)
    pred_list, obs_list = model_run.test_dataloader(model, testloader, seq_first=True)
    pred = reduce(lambda x, y: np.vstack((x, y)), pred_list)
    obs = reduce(lambda x, y: np.vstack((x, y)), obs_list)
    stat_dict = data_model.stat_dict
    # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
    pred = _trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
    obs = _trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)
    return pred, obs


def master_train_1by1(data_model, valid_size=0.2):
    model_dict = data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_loss = model_dict['loss']
    opt_train = model_dict['train']

    # data
    opt_model['nx'] = data_model.data_forcing.shape[-1]  # + data_model.data_attr.shape[-1]
    opt_model['ny'] = 1
    # loss
    if opt_loss['name'] == 'RmseLoss':
        loss_fun = crit.RmseLoss()
    elif opt_loss['name'] == 'NSELosstest':
        loss_fun = crit.NSELosstest()
    elif opt_loss['name'] == 'NSELoss':
        loss_fun = crit.NSELoss()
    else:
        print("Please specify the loss function!!!")

    # model
    if opt_model['name'] == 'CudnnLstmModel':
        model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])
    elif opt_model['name'] == 'LstmCloseModel':
        model = rnn.LstmCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                   fillObs=True)
    elif opt_model['name'] == 'AnnModel':
        model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'])
    elif opt_model['name'] == 'AnnCloseModel':
        model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                  fillObs=True)

    # train model
    output_dir = model_dict['dir']['Out']
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    trainloader, validloader = create_datasets(data_model, valid_size)
    model, avg_train_losses, avg_valid_losses = model_run.train_valid_dataloader(model, trainloader, validloader,
                                                                                 loss_fun, opt_train['nEpoch'],
                                                                                 output_dir, opt_train['saveEpoch'])
    return model, avg_train_losses, avg_valid_losses


def master_train(data_model, valid_size=0, pre_trained_model_epoch=1):
    model_dict = data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_loss = model_dict['loss']
    opt_train = model_dict['train']

    # data
    x, y, c = data_model.load_data(model_dict)
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]
    opt_model['nx'] = nx
    opt_model['ny'] = ny
    # loss
    if opt_loss['name'] == 'SigmaLoss':
        loss_fun = crit.SigmaLoss(prior=opt_loss['prior'])
        opt_model['ny'] = ny * 2
    elif opt_loss['name'] == 'RmseLoss':
        loss_fun = crit.RmseLoss()
        opt_model['ny'] = ny
    elif opt_loss['name'] == 'NSELosstest':
        loss_fun = crit.NSELosstest()
        opt_model['ny'] = ny
    elif opt_loss['name'] == 'NSELoss':
        loss_fun = crit.NSELoss()
        opt_model['ny'] = ny
    else:
        print("Please specify the loss function!!!")

    # model
    if opt_train['saveEpoch'] > opt_train['nEpoch']:
        opt_train['saveEpoch'] = opt_train['nEpoch']
    out = model_dict['dir']['Out']
    if not os.path.isdir(out):
        os.makedirs(out)
    if pre_trained_model_epoch > 1:
        pre_trained_model_file = os.path.join(out, 'model_Ep' + str(pre_trained_model_epoch) + '.pt')
        model = rnn.CudnnLstmModelPretrain(nx=opt_model['nx'], ny=opt_model['ny'],
                                           hidden_size=opt_model['hiddenSize'],
                                           pretrian_model_file=pre_trained_model_file)
    else:
        if opt_model['name'] == 'CudnnLstmModel':
            model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])
        elif opt_model['name'] == 'LstmCloseModel':
            model = rnn.LstmCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                       fillObs=True)
        elif opt_model['name'] == 'AnnModel':
            model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'])
        elif opt_model['name'] == 'AnnCloseModel':
            model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                      fillObs=True)

    # train model
    if valid_size > 0:
        model, train_loss, valid_loss = model_run.model_train_valid(model, x, y, c, loss_fun,
                                                                    n_epoch=opt_train['nEpoch'],
                                                                    mini_batch=opt_train['miniBatch'],
                                                                    save_epoch=opt_train['saveEpoch'],
                                                                    save_folder=out, valid_size=valid_size)
        return model, train_loss, valid_loss
    else:
        model = model_run.model_train(model, x, y, c, loss_fun, n_epoch=opt_train['nEpoch'],
                                      mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'],
                                      save_folder=out, pre_trained_model_epoch=pre_trained_model_epoch)
        return model


def master_test(data_model, epoch=-1):
    """:parameter
        data_model：测试使用的数据
        model_dict：测试时的模型配置
    """
    model_dict = data_model.data_source.data_config.model_dict
    opt_data = model_dict['data']
    opt_model = model_dict['model']
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    x, obs, c = data_model.load_data(model_dict)

    # generate file names and run model
    out = model_dict['dir']['Out']
    t_range = data_model.t_s_dict["t_final_range"]
    if epoch < 0:
        epoch = model_dict['train']["nEpoch"]
    model_file = os.path.join(out, 'model_Ep' + str(epoch) + '.pt')
    file_path = name_pred(model_dict, out, t_range, epoch)
    print('output files:', file_path)
    if not os.path.isfile(model_file):
        model_file = os.path.join(out, 'checkpoint.pt')
        opt_model['nx'] = x.shape[-1] + c.shape[-1]
        opt_model['ny'] = obs.shape[-1]
        if opt_model['name'] == 'CudnnLstmModel':
            model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])
        elif opt_model['name'] == 'LstmCloseModel':
            model = rnn.LstmCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                       fillObs=True)
        elif opt_model['name'] == 'AnnModel':
            model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'])
        elif opt_model['name'] == 'AnnCloseModel':
            model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                      fillObs=True)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        model_run.model_test_valid(model, x, c, file_path=file_path, batch_size=batch_size)
    else:
        # 如果没有测试结果，那么就重新运行测试代码
        re_test = False
        if not os.path.isfile(file_path):
            re_test = True
        if re_test:
            print('Runing new results')
            model = torch.load(model_file)
            model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
        else:
            print('Loaded previous results')

    # load previous result并反归一化为标准量纲
    data_pred = pd.read_csv(file_path, dtype=np.float, header=None).values
    is_sigma_x = False
    if model_dict['loss']['name'] == 'SigmaLoss':
        # TODO：sigmaloss下的情况都没做
        is_sigma_x = True
        pred = data_pred[:, :, ::2]
        sigma_x = data_pred[:, :, 1::2]
    else:
        # 扩充到三维才能很好地在后面调用stat.trans_norm函数反归一化
        pred = np.expand_dims(data_pred, axis=2)
    if opt_data['doNorm'][1] is True:
        stat_dict = data_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = _trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        obs = _trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)

    if is_sigma_x is True:
        return pred, obs, sigma_x
    else:
        return pred, obs


def master_train_easy_lstm(data_model):
    """training main function for stacked lstm"""
    model_dict = data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_loss = model_dict['loss']
    opt_train = model_dict['train']

    # data
    x, y, c = data_model.load_data(model_dict)
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]
    opt_model['nx'] = nx
    opt_model['ny'] = ny
    # loss
    if opt_loss['name'] == 'SigmaLoss':
        loss_fun = crit.SigmaLoss(prior=opt_loss['prior'])
        opt_model['ny'] = ny * 2
    elif opt_loss['name'] == 'RmseLoss':
        loss_fun = crit.RmseLoss()
        opt_model['ny'] = ny
    elif opt_loss['name'] == 'NSELosstest':
        loss_fun = crit.NSELosstest()
        opt_model['ny'] = ny
    elif opt_loss['name'] == 'NSELoss':
        loss_fun = crit.NSELoss()
        opt_model['ny'] = ny
    else:
        print("Please specify the loss function!!!")

    # model
    if opt_model['name'] == 'LinearEasyLstm':
        model = easy_lstm.LinearEasyLstm(nx=opt_model['nx'], ny=opt_model['ny'],
                                         hidden_size=opt_model['hiddenSize'])
    elif opt_model['name'] == 'StackedEasyLstm':
        model = easy_lstm.StackedEasyLstm(nx=opt_model['nx'], ny=opt_model['ny'],
                                          hidden_size=opt_model['hiddenSize'])
    elif opt_model['name'] == 'PytorchLstm':
        model = easy_lstm.PytorchLstm(nx=opt_model['nx'], ny=opt_model['ny'],
                                      hidden_size=opt_model['hiddenSize'])
    else:
        model = easy_lstm.EasyLstm(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])

    # train model
    output_dir = model_dict['dir']['Out']
    model_run.model_train_easy_lstm(model, x, y, c, loss_fun, n_epoch=opt_train['nEpoch'],
                                    mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'],
                                    save_folder=output_dir)


def master_test_easy_lstm(data_model, load_epoch=-1):
    """data_model：测试使用的数据 ;load_epoch：the loaded model's epoch"""
    model_dict = data_model.data_source.data_config.model_dict
    opt_data = model_dict['data']
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    x, obs, c = data_model.load_data(model_dict)

    # generate file names and run model
    out = model_dict['dir']['Out']
    t_range = data_model.t_s_dict["t_final_range"]
    if load_epoch < 0:
        load_epoch = model_dict['train']["nEpoch"]
    file_path = name_pred(model_dict, out, t_range, load_epoch)
    print('output files:', file_path)

    model = model_run.model_load(out, load_epoch)
    model_run.model_test_easy_lstm(model, x, c, file_path=file_path, batch_size=batch_size)

    data_pred = pd.read_csv(file_path, dtype=np.float, header=None).values
    # 扩充到三维才能很好地在后面调用stat.trans_norm函数反归一化
    pred = np.expand_dims(data_pred, axis=2)
    if opt_data['doNorm'][1] is True:
        stat_dict = data_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = _trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        obs = _trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)
    return pred, obs


def master_train_better_lstm(dataset):
    model_dict = dataset.data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_loss = model_dict['loss']
    opt_train = model_dict['train']

    # data
    trainloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True)
    opt_model['nx'] = dataset.xc.shape[-1]
    opt_model['ny'] = dataset.y.shape[-1]
    # loss
    if opt_loss['name'] == 'RmseLoss':
        loss_fun = crit.RmseLoss()
    elif opt_loss['name'] == 'NSELosstest':
        loss_fun = crit.NSELosstest()
    elif opt_loss['name'] == 'NSELoss':
        loss_fun = crit.NSELoss()
    else:
        print("Please specify the loss function!!!")

    # model
    if opt_model['name'] == 'LinearEasyLstm':
        model = easy_lstm.LinearEasyLstm(nx=opt_model['nx'], ny=opt_model['ny'],
                                         hidden_size=opt_model['hiddenSize'])
    elif opt_model['name'] == 'StackedEasyLstm':
        model = easy_lstm.StackedEasyLstm(nx=opt_model['nx'], ny=opt_model['ny'],
                                          hidden_size=opt_model['hiddenSize'])
    elif opt_model['name'] == 'PytorchLstm':
        model = easy_lstm.PytorchLstm(nx=opt_model['nx'], ny=opt_model['ny'],
                                      hidden_size=opt_model['hiddenSize'])
    else:
        model = easy_lstm.EasyLstm(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])

    # train model
    output_dir = model_dict['dir']['Out']
    model_save_dir = os.path.join(output_dir, 'model')
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    model_run.train_dataloader(model, trainloader, loss_fun, opt_train['nEpoch'], output_dir, model_save_dir,
                               opt_train['saveEpoch'])


def master_test_better_lstm(dataset, load_epoch=-1):
    model_dict = dataset.data_model.data_source.data_config.model_dict
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    # data
    testloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=False)

    # model
    out_folder = model_dict['dir']['Out']
    opt_train = model_dict['train']
    if load_epoch < 0:
        load_epoch = opt_train['nEpoch']
    model_file = os.path.join(out_folder, 'model', 'model' + '_Ep' + str(load_epoch) + '.pt')
    model = torch.load(model_file)
    pred_list, obs_list = model_run.test_dataloader(model, testloader)
    pred = reduce(lambda x, y: np.vstack((x, y)), pred_list)
    obs = reduce(lambda x, y: np.vstack((x, y)), obs_list)
    # 扩充到三维才能很好地在后面调用stat.trans_norm函数反归一化
    stat_dict = dataset.data_model.stat_dict
    # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
    pred = _trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
    obs = _trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)

    return pred, obs


def master_train_natural_flow(model_input, pre_trained_model_epoch=1):
    model_dict = model_input.data_model2.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_train = model_dict['train']

    # data
    x, y, c = model_input.load_data(model_dict)
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
    if pre_trained_model_epoch > 1:
        pre_trained_model_file = os.path.join(out, 'model_Ep' + str(pre_trained_model_epoch) + '.pt')
        model = rnn.CudnnLstmModelPretrain(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'],
                                           pretrian_model_file=pre_trained_model_file)
    else:
        model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])

    # train model
    model_run.model_train(model, x, y, c, loss_fun, n_epoch=opt_train['nEpoch'],
                          mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'], save_folder=out,
                          pre_trained_model_epoch=pre_trained_model_epoch)


def master_test_natural_flow(model_input, epoch=-1):
    """:parameter
        data_model：测试使用的数据
        model_dict：测试时的模型配置
    """
    data_model = model_input.data_model2
    model_dict = data_model.data_source.data_config.model_dict
    opt_data = model_dict['data']
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    x, obs, c = model_input.load_data(model_dict)

    # generate file names and run model
    out = os.path.join(model_dict['dir']['Out'], "model")
    t_range = data_model.t_s_dict["t_final_range"]
    if epoch < 0:
        epoch = model_dict['train']["nEpoch"]
    file_path = name_pred(model_dict, out, t_range, epoch)
    print('output files:', file_path)
    # 如果没有测试结果，那么就重新运行测试代码
    re_test = False
    if not os.path.isfile(file_path):
        re_test = True
    if re_test:
        print('Runing new results')
        model = model_run.model_load(out, epoch)
        model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
    else:
        print('Loaded previous results')

    # load previous result并反归一化为标准量纲
    data_pred = pd.read_csv(file_path, dtype=np.float, header=None).values

    # 扩充到三维才能很好地在后面调用stat.trans_norm函数反归一化
    pred = np.expand_dims(data_pred, axis=2)
    if opt_data['doNorm'][1] is True:
        stat_dict = data_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = _trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        obs = _trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)

    return pred, obs


def train_lstm_inv(data_model, pre_trained_model_epoch=1):
    """call lstm inv model to train"""
    model_dict = data_model.model_dict1
    opt_model = model_dict['model']
    opt_train = model_dict['train']

    # data
    xqch, xct, qt = data_model.load_data()
    theta_length = 10
    opt_model['nx'] = (xqch.shape[-1], xct.shape[-1], theta_length)
    opt_model['ny'] = qt.shape[-1]
    # loss
    loss_fun = crit.RmseLoss()
    # model
    output_dir = model_dict['dir']['Out']
    if pre_trained_model_epoch > 1:
        pre_trained_model_file = os.path.join(output_dir, 'model_Ep' + str(pre_trained_model_epoch) + '.pt')
        model_inv = rnn.CudnnLstmModelInvKernelPretrain(nx=opt_model['nx'], ny=opt_model['ny'],
                                                        hidden_size=opt_model['hiddenSize'],
                                                        pretrian_model_file=pre_trained_model_file)
    else:
        if opt_model['name'] == 'CudnnLstmModelInv':
            model_inv = rnn.CudnnLstmModelInv(nx=opt_model['nx'], ny=opt_model['ny'],
                                              hidden_size=opt_model['hiddenSize'])
        else:
            model_inv = rnn.CudnnLstmModelInvKernel(nx=opt_model['nx'], ny=opt_model['ny'],
                                                    hidden_size=opt_model['hiddenSize'])

    # train model
    model_run.model_train_inv(model_inv, xqch, xct, qt, loss_fun, n_epoch=opt_train['nEpoch'],
                              mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'],
                              save_folder=output_dir, pre_trained_model_epoch=pre_trained_model_epoch)


def test_lstm_inv(data_model, epoch=-1):
    model_dict = data_model.model_dict1
    opt_data = model_dict['data']
    opt_model = model_dict['model']
    opt_train = model_dict['train']
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    # data
    xqch, xct, qt = data_model.load_data()
    theta_length = 10
    # generate file names and run model
    out = os.path.join(model_dict['dir']['Out'])
    t_range = data_model.model_dict2["data"]["tRangeTest"]
    if epoch < 0:
        epoch = opt_train["nEpoch"]
    file_path = name_pred(model_dict, out, t_range, epoch)
    print('output files:', file_path)
    model = model_run.model_load(out, epoch)
    if opt_model["name"] == "CudnnLstmModelInv":
        data_pred, data_params = model_run.model_test_inv(model, xqch, xct, batch_size)
    else:
        data_pred, data_params = model_run.model_test_inv_kernel(model, xqch, xct, batch_size)

    data_stack = reduce(lambda a, b: np.vstack((a, b)),
                        list(map(lambda x: x.reshape(x.shape[0], x.shape[1]), data_pred)))
    pred = np.expand_dims(data_stack, axis=2)
    if opt_data['doNorm'][1] is True:
        stat_dict = data_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = _trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        qt = _trans_norm(qt, 'usgsFlow', stat_dict, to_norm=False)

    return pred, qt


def train_lstm_siminv(data_input, pre_trained_model_epoch=1):
    model_dict = data_input.lstm_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_train = model_dict['train']

    # data
    xqqnch, xqnct, qt = data_input.load_data()
    theta_length = 10
    opt_model['nx'] = (xqqnch.shape[-1], xqnct.shape[-1], theta_length)
    opt_model['ny'] = qt.shape[-1]
    # loss
    loss_fun = crit.RmseLoss()
    # model
    output_dir = model_dict['dir']['Out']
    model_save_dir = os.path.join(output_dir, 'model')
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    if pre_trained_model_epoch > 1:
        pre_trained_model_file = os.path.join(model_save_dir, 'model_Ep' + str(pre_trained_model_epoch) + '.pt')
        model_inv = rnn.CudnnLstmModelInvKernelPretrain(nx=opt_model['nx'], ny=opt_model['ny'],
                                                        hidden_size=opt_model['hiddenSize'],
                                                        pretrian_model_file=pre_trained_model_file)
    else:
        if opt_model['name'] == 'CudnnLstmModelInv':
            model_inv = rnn.CudnnLstmModelInv(nx=opt_model['nx'], ny=opt_model['ny'],
                                              hidden_size=opt_model['hiddenSize'])
        else:
            model_inv = rnn.CudnnLstmModelInvKernel(nx=opt_model['nx'], ny=opt_model['ny'],
                                                    hidden_size=opt_model['hiddenSize'])
    # train model
    model_run.model_train_inv(model_inv, xqqnch, xqnct, qt, loss_fun, n_epoch=opt_train['nEpoch'],
                              mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'],
                              save_folder=model_save_dir, pre_trained_model_epoch=pre_trained_model_epoch)


def test_lstm_siminv(data_input, epoch=-1):
    model_dict = data_input.lstm_model.data_source.data_config.model_dict
    opt_data = model_dict['data']
    opt_model = model_dict['model']
    opt_train = model_dict['train']
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    # data
    xqqnch, xqnct, qt = data_input.load_data()
    theta_length = 10
    # generate file names and run model
    out = os.path.join(model_dict['dir']['Out'], 'model')
    t_range = data_input.lstm_model.t_s_dict["t_final_range"]
    if epoch < 0:
        epoch = opt_train["nEpoch"]
    file_path = name_pred(model_dict, out, t_range, epoch)
    print('output files:', file_path)
    model = model_run.model_load(out, epoch)
    if opt_model["name"] == "CudnnLstmModelInv":
        data_pred, data_params = model_run.model_test_inv(model, xqqnch, xqnct, batch_size)
    else:
        data_pred, data_params = model_run.model_test_inv_kernel(model, xqqnch, xqnct, batch_size)

    data_stack = reduce(lambda a, b: np.vstack((a, b)),
                        list(map(lambda x: x.reshape(x.shape[0], x.shape[1]), data_pred)))
    pred = np.expand_dims(data_stack, axis=2)
    if opt_data['doNorm'][1] is True:
        stat_dict = data_input.lstm_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = _trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        qt = _trans_norm(qt, 'usgsFlow', stat_dict, to_norm=False)

    return pred, qt


def train_lstm_da(data_input, pre_trained_model_epoch=1):
    model_dict = data_input.data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_train = model_dict['train']

    # data
    qx, y, c = data_input.load_data(model_dict)
    opt_model['nx'] = qx.shape[-1] + c.shape[-1]
    opt_model['ny'] = y.shape[-1]
    # loss
    loss_fun = crit.RmseLoss()
    # model
    output_dir = model_dict['dir']['Out']
    if pre_trained_model_epoch > 1:
        pre_trained_model_file = os.path.join(output_dir, 'model_Ep' + str(pre_trained_model_epoch) + '.pt')
        model = rnn.CudnnLstmModelPretrain(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'],
                                           pretrian_model_file=pre_trained_model_file)
    else:
        model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])
    # train model
    model_run.model_train(model, qx, y, c, loss_fun, n_epoch=opt_train['nEpoch'],
                          mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'],
                          save_folder=output_dir, pre_trained_model_epoch=pre_trained_model_epoch)


def test_lstm_da(data_input, epoch=-1):
    model_dict = data_input.data_model.data_source.data_config.model_dict
    opt_data = model_dict['data']
    opt_model = model_dict['model']
    opt_train = model_dict['train']
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    # data
    qx, obs, c = data_input.load_data(model_dict)
    # generate file names and run model
    out = model_dict['dir']['Out']
    t_range = data_input.data_model.t_s_dict["t_final_range"]
    if epoch < 0:
        epoch = opt_train["nEpoch"]
    file_path = name_pred(model_dict, out, t_range, epoch)
    print('output files:', file_path)
    model = model_run.model_load(out, epoch)

    model_run.model_test(model, qx, c, file_path=file_path, batch_size=batch_size)
    # load previous result并反归一化为标准量纲
    data_pred = pd.read_csv(file_path, dtype=np.float, header=None).values

    # 扩充到三维才能很好地在后面调用stat.trans_norm函数反归一化
    pred = np.expand_dims(data_pred, axis=2)
    if opt_data['doNorm'][1] is True:
        stat_dict = data_input.data_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = _trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        obs = _trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)

    return pred, obs


def train_lstm_forecast(data_input):
    data_model = data_input.model_data
    model_dict = data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_train = model_dict['train']

    # data
    x, y, c = data_input.load_data(model_dict)
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]
    opt_model['nx'] = nx
    opt_model['ny'] = ny
    # loss
    loss_fun = crit.RmseLoss()
    # model
    model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])
    # train model
    out = os.path.join(model_dict['dir']['Out'], "model")
    if not os.path.isdir(out):
        os.mkdir(out)
    model_run.model_train(model, x, y, c, loss_fun, n_epoch=opt_train['nEpoch'],
                          mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'], save_folder=out)


def test_lstm_forecast(data_input):
    data_model = data_input.model_data
    model_dict = data_model.data_source.data_config.model_dict
    opt_data = model_dict['data']
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    x, obs, c = data_input.load_data(model_dict)

    # generate file names and run model
    out = os.path.join(model_dict['dir']['Out'], "model")
    t_range = data_model.t_s_dict["t_final_range"]
    epoch = model_dict['train']["nEpoch"]
    file_path = name_pred(model_dict, out, t_range, epoch)
    print('output files:', file_path)
    # 如果没有测试结果，那么就重新运行测试代码
    re_test = False
    if not os.path.isfile(file_path):
        re_test = True
    if re_test:
        print('Runing new results')
        model = model_run.model_load(out, epoch)
        model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
    else:
        print('Loaded previous results')

    # load previous result并反归一化为标准量纲
    data_pred = pd.read_csv(file_path, dtype=np.float, header=None).values

    # 扩充到三维才能很好地在后面调用stat.trans_norm函数反归一化
    pred = np.expand_dims(data_pred, axis=2)
    if opt_data['doNorm'][1] is True:
        stat_dict = data_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = stat.trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        obs = stat.trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)

    return pred, obs
