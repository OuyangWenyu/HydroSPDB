from abc import ABC, abstractmethod
from typing import Dict
import torch
import json
import os
from datetime import datetime

from data.data_base import DatasetBase
from data.data_dict import dataloaders_dict
from hydroDL.model_dict_function import pytorch_model_dict


class TimeSeriesModel(ABC):
    """
    An abstract class used to handle different configurations
    of models + hyperparams for training, test, and predict functions.
    This class assumes that data is already split into test train
    and validation at this point.
    """

    def __init__(
            self,
            model_base: str,
            dataset_model: DatasetBase,
            params: Dict):
        """
        Parameters:
            model_base: name of the model
            dataset_model: the digital twin of a dataset in reality
            params: the configuration parameters
        """
        self.params = params
        if "weight_path" in params["model_params"]:
            self.model = self.load_model(model_base, params["model_params"], params["model_params"]["weight_path"])
        else:
            self.model = self.load_model(model_base, params["model_params"])
        self.training = self.make_data_load(dataset_model, params["dataset_params"], "train")
        if params["dataset_params"]["t_range_valid"] is not None:
            self.validation = self.make_data_load(dataset_model, params["dataset_params"], "valid")
        self.test_data = self.make_data_load(dataset_model, params["dataset_params"], "test")

    @abstractmethod
    def load_model(self, model_base: str, model_params: Dict, weight_path=None) -> object:
        """
        This function should load and return the model
        this will vary based on the underlying framework used
        """
        raise NotImplementedError

    @abstractmethod
    def make_data_load(self, dataset_model: DatasetBase, params: Dict, loader_type: str) -> object:
        """
        Intializes a data loader based on the provided data_path.
        This may be as simple as a pandas dataframe or as complex as
        a custom PyTorch data loader.
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, output_path: str, epoch: int):
        """
        Saves a model to a specific path along with a configuration report
        of the parameters and data info.
        """
        raise NotImplementedError


class PyTorchForecast(TimeSeriesModel):
    def __init__(
            self,
            model_base: str,
            dataset_model: DatasetBase,
            params_dict: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(model_base, dataset_model, params_dict)
        print("Torch is using " + str(self.device))

    def load_model(self, model_base: str, model_params: Dict, weight_path: str = None, strict=True):
        if model_base in pytorch_model_dict:
            model = pytorch_model_dict[model_base](**model_params["model_param"])
            if weight_path:
                # if the model has been trained
                strict = False
                checkpoint = torch.load(weight_path, map_location=self.device)
                if "weight_path_add" in model_params:
                    if "excluded_layers" in model_params["weight_path_add"]:
                        excluded_layers = model_params["weight_path_add"]["excluded_layers"]
                        for layer in excluded_layers:
                            del checkpoint[layer]
                        print("sucessfully deleted layers")
                    else:
                        print("directly loading identically-named layers of source model and no freeze")
                model.load_state_dict(checkpoint, strict=strict)
                print("Weights sucessfully loaded")
            model.to(self.device)
            # if need to freeze some layers from the source model when transfer learning
            if weight_path:
                if "weight_path_add" in model_params:
                    if "freeze_params" in model_params["weight_path_add"]:
                        freeze_params = model_params["weight_path_add"]["freeze_params"]
                        for param in freeze_params:
                            exec("model." + param + ".requires_grad = False")
        else:
            raise Exception("Error the model " + model_base + " was not found in the model dict. Please add it.")
        return model

    def save_model(self, final_path: str, epoch: int) -> None:
        """
        Function to save a model to a given file path
        """
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        time_stamp = datetime.now().strftime("%d_%B_%Y%I_%M%p")
        model_name = time_stamp + "_model.pth"
        params_name = time_stamp + ".json"
        model_save_path = os.path.join(final_path, model_name)
        params_save_path = os.path.join(final_path, time_stamp + ".json")
        torch.save(self.model.state_dict(), model_save_path)
        with open(params_save_path, "w+") as p:
            json.dump(self.params, p)

    def make_data_load(self, dataset_model: DatasetBase, dataset_params: Dict, loader_type: str, the_class="default"):
        the_loader = dataset_params["data_loader"]
        if the_loader in list(dataloaders_dict.keys()):
            loader = dataloaders_dict[the_loader](dataset_model, dataset_params, loader_type)
        else:
            raise Exception(
                "Error the data model " + str(dataset_model) + " was not found in the model dict. Please add it.")
        return loader
