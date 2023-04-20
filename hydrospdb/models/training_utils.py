"""
Author: Wenyu Ouyang
Date: 2021-08-09 10:19:13
LastEditTime: 2023-04-20 17:54:38
LastEditors: Wenyu Ouyang
Description: Some util classes and functions during hydroDL training or testing
FilePath: /HydroSPDB/hydrospdb/models/training_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from typing import Union
import torch
from hydrospdb.utils.hydro_utils import hydro_logger


def get_the_device(device_num: Union[list, int]):
    """
    Get device for torch according to its name

    Parameters
    ----------
    device_num : Union[list, int]
        number of the device -- -1 means "cpu" or 0, 1, ... means "cuda:x"
    """
    if device_num == [-1] or device_num == -1 or device_num == ["-1"]:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if device_num != [-1] and device_num != -1 and device_num != ["-1"]:
            hydro_logger.warning("You don't have GPU, so have to choose cpu for models")
        return torch.device("cpu")
    else:
        if type(device_num) is not list:
            return torch.device("cuda:" + str(device_num))
        else:
            # when using multiple GPUs, we also set a device for model
            # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
            return torch.device("cuda:" + str(device_num[0]))


class EarlyStopper(object):
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        """
        EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

        Parameters
        ----------
        patience
            Number of events to wait if no improvement and then stop the training.
        min_delta
            A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta
            It True, `min_delta` defines an increase since the last `patience` reset, otherwise,
        it defines an increase after the last event. Default value is False.
        """

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None

    def check_loss(self, model, validation_loss) -> bool:
        score = validation_loss
        if self.best_score is None:
            self.save_model_checkpoint(model)
            self.best_score = score

        elif score + self.min_delta >= self.best_score:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            print(self.counter)
            if self.counter >= self.patience:
                return False
        else:
            self.save_model_checkpoint(model)
            self.best_score = score
            self.counter = 0
        return True

    def save_model_checkpoint(self, model):
        torch.save(model.state_dict(), "checkpoint.pth")
