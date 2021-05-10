import os
import unittest
from datetime import datetime

from hydroDL.explain_model_output import (
    deep_explain_model_heatmap,
    deep_explain_model_summary_plot,
)
from hydroDL.preprocessing.pytorch_loaders import CSVTestLoader
from hydroDL.time_model import PyTorchForecast


class ModelInterpretabilityTest(unittest.TestCase):
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
    test_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    model_params: dict = {
        "model_name": "MultiAttnHeadSimple",
        "model_params": {"number_time_series": 3, "seq_len": 20, "output_seq_len": 10},
        "metrics": ["MSE", "DilateLoss"],
        "dataset_params": {
            "forecast_history": 20,
            "class": "default",
            "forecast_length": 10,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate": False,
        },
        "wandb": {
            "name": "flood_forecast_circleci",
            "tags": ["dummy_run", "circleci"],
            "project": "repo-hydroDL",
        },
        "inference_params": {
            "hours_to_forecast": 30,
            "datetime_start": datetime(2014, 6, 2, 0),
        },
    }
    lstm_model_params: dict = {
        "model_name": "LSTM",
        "model_params": {"seq_length": 20, "n_time_series": 3, "output_seq_len": 10},
        "metrics": ["MSE", "DilateLoss"],
        "dataset_params": {
            "forecast_history": 20,
            "class": "default",
            "forecast_length": 10,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate": False,
        },
        "wandb": {
            "name": "flood_forecast_circleci",
            "tags": ["dummy_run", "circleci"],
            "project": "repo-hydroDL",
        },
        "inference_params": {
            "hours_to_forecast": 30,
            "datetime_start": datetime(2014, 6, 2, 0),
        },
    }
    simple_param = {
        "model_name": "SimpleLinearModel",
        "metrics": ["MSE", "DilateLoss"],
        "use_decoder": True,
        "model_params": {"n_time_series": 3, "seq_length": 20, "output_seq_len": 10},
        "dataset_params": {
            "forecast_history": 20,
            "class": "default",
            "forecast_length": 10,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate": False,
            "train_end": 50,
            "valid_end": 100,
        },
        "inference_params": {
            "hours_to_forecast": 30,
            "datetime_start": datetime(2014, 6, 2, 0),
        },
        "training_params": {
            "optimizer": "Adam",
            "lr": 0.01,
            "criterion": "MSE",
            "epochs": 1,
            "batch_size": 2,
            "optim_params": {},
        },
        "wandb": False,
    }
    keag_file = os.path.join(test_path, "keag_small.csv")
    model = PyTorchForecast(
        "MultiAttnHeadSimple", keag_file, keag_file, keag_file, model_params
    )
    lstm_model = PyTorchForecast(
        "LSTM", keag_file, keag_file, keag_file, lstm_model_params
    )
    simple_linear_model = PyTorchForecast(
        "SimpleLinearModel", keag_file, keag_file, keag_file, simple_param
    )
    data_base_params = {
        "file_path": os.path.join(test_path2, "keag_small.csv"),
        "forecast_history": 20,
        "forecast_length": 10,
        "relevant_cols": ["cfs", "temp", "precip"],
        "target_col": ["cfs"],
        "interpolate_param": False,
    }
    csv_test_loader = CSVTestLoader(
        df_path=os.path.join(test_path2, "keag_small.csv"),
        forecast_total=model_params["inference_params"]["hours_to_forecast"],
        **data_base_params
    )

    def test_deep_explain_model_summary_plot(self):
        deep_explain_model_summary_plot(
            model=self.model, csv_test_loader=self.csv_test_loader
        )
        deep_explain_model_summary_plot(
            model=self.lstm_model, csv_test_loader=self.csv_test_loader
        )
        deep_explain_model_summary_plot(
            model=self.simple_linear_model, csv_test_loader=self.csv_test_loader
        )
        # dummy assert
        self.assertEqual(1, 1)

    def test_deep_explain_model_heatmap(self):
        deep_explain_model_heatmap(
            model=self.model, csv_test_loader=self.csv_test_loader
        )
        deep_explain_model_heatmap(
            model=self.lstm_model, csv_test_loader=self.csv_test_loader
        )
        deep_explain_model_heatmap(
            model=self.simple_linear_model, csv_test_loader=self.csv_test_loader
        )
        # dummy assert
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
