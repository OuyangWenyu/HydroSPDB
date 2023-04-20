"""
Author: Wenyu Ouyang
Date: 2023-04-06 14:45:34
LastEditTime: 2023-04-20 15:32:58
LastEditors: Wenyu Ouyang
Description: Test the multioutput model
FilePath: /HydroSPDB/tests/test_spdb.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import torch
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))


def test_cuda_available():
    assert torch.cuda.is_available()
