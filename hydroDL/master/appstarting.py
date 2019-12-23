# run training script in detached screen
# careful! setup python environment before script running

# bash
# source /home/kxf227/anaconda3/bin/activate
# conda activate pytorch

import os
import argparse

import data.data_input
import data.read_config
from hydroDL import master
from utils import send_email


def run_train(master_dict, *, screen='test', cuda_id):
    """后台运行"""
    if type(master_dict) is str:
        m_file = master_dict
        master_dict = data.read_config.read_master_file(m_file)
    else:
        m_file = data.read_config.write_master_file(master_dict)

    code_path = os.path.realpath(__file__)
    if screen is None:
        cmd = 'CUDA_VISIBLE_DEVICES={} python {} -F {} -M {}'.format(
            cuda_id, code_path, 'train', m_file)
    else:
        cmd = 'CUDA_VISIBLE_DEVICES={} screen -dmS {} python {} -F {} -M {}'.format(
            cuda_id, screen, code_path, 'train', m_file)

    print(cmd)

    parser = argparse.ArgumentParser()
    parser.add_argument('-F', dest='func', type=str, default='train')
    parser.add_argument('-M', dest='m_file', type=str, default=m_file)
    args = parser.parse_args()
    if args.func == 'train':
        m_dict = data.read_config.read_master_file(args.m_file)
        master.master_train(m_dict)
        out = m_dict['out']
        send_email.send_email(subject='Training Done', text=out)
    # os.system(cmd)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-F', dest='func', type=str)
#     parser.add_argument('-M', dest='mFile', type=str)
#     args = parser.parse_args()
#     if args.func == 'train':
#         mDict = master.readMasterFile(args.mFile)
#         master.train(mDict)
#         out = mDict['out']
#         email.sendEmail(subject='Training Done', text=out)
