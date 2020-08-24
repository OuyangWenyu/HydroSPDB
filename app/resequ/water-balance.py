"""physics-guided deep learning for streamflow prediction in basins with reservoir"""
import argparse
import os
import sys
from easydict import EasyDict as edict
import numpy as np
import torch

sys.path.append("../..")
from data.config import cmd
from data import GagesConfig, GagesSource
from data.data_input import GagesModel, save_datamodel, _basin_norm, save_result
from hydroDL import master_train, master_test


def pgml(args):
    print(args)


if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    pgml(args)
    print("End\n")
