import os
import shutil
import time
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import definitions
from utils.hydro_utils import progress_wrapped, provide_progress_bar


def long_running_function(*args, **kwargs):
    # print("Running with args:%s and kwargs:%s" % (args, kwargs))
    time.sleep(5)
    return "success"


@progress_wrapped(estimated_time=5)
def another_long_running_function(*args, **kwargs):
    # print("Running with args:%s and kwargs:%s" % (args, kwargs))
    time.sleep(5)
    return "success"


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print("test util functions...")

    def test_priestley_taylor_pet(self):
        print("test priestley_taylor equations")

    def test_progress_bar(self):
        # Basic example
        retval = provide_progress_bar(long_running_function, estimated_time=5)
        print(retval)

        # Full example
        retval = provide_progress_bar(long_running_function,
                                      estimated_time=5, tstep=1 / 5.0,
                                      tqdm_kwargs={
                                          "bar_format": '{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}]'},
                                      args=(1, "foo"), kwargs={"spam": "eggs"}
                                      )
        print(retval)

        # Example of using the decorator
        retval = another_long_running_function()
        print(retval)

    def test_gpu(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # cuda is TITAN
        x = torch.tensor([1., 2.]).cuda()
        # x.device is device(type='cuda', index=0)
        y = torch.tensor([1., 2.]).cuda()
        print(x)
        print(y)

    def test_dataparallel(self):
        # Parameters and DataLoaders
        input_size = 5
        output_size = 2

        batch_size = 30
        data_size = 100
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class RandomDataset(Dataset):

            def __init__(self, size, length):
                self.len = length
                self.data = torch.randn(length, size)

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return self.len

        rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                                 batch_size=batch_size, shuffle=True)

        class Model(nn.Module):
            # Our model

            def __init__(self, input_size, output_size):
                super(Model, self).__init__()
                self.fc = nn.Linear(input_size, output_size)

            def forward(self, input):
                output = self.fc(input)
                print("\tIn Model: input size", input.size(),
                      "output size", output.size())

                return output

        model = Model(input_size, output_size)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            # model = nn.DataParallel(model)
            model = nn.DataParallel(model, device_ids=[0, 1])

        model.to(device)
        for data in rand_loader:
            input = data.to(device)
            output = model(input)
            print("Outside: input size", input.size(),
                  "output_size", output.size())


if __name__ == '__main__':
    unittest.main()
