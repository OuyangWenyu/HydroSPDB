from typing import Callable, Dict, List, Tuple, Type
from functools import reduce
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.utils.data import DataLoader, Dataset
from data.data_loaders import TestDataModel
from visual.explain_model_output import (
    deep_explain_model_heatmap,
    deep_explain_model_summary_plot,
)
from hydroDL.model_dict_function import sequence_first_model_lst
from hydroDL.time_model import TimeSeriesModel
from explore.stat import statError


def stream_baseline(river_flow_df: pd.DataFrame, forecast_column: str, hours_forecast=336) -> (pd.DataFrame, float):
    """
    Function to compute the baseline MSE
    by using the mean value from the train data.
    """
    total_length = len(river_flow_df.index)
    train_river_data = river_flow_df[: total_length - hours_forecast]
    test_river_data = river_flow_df[total_length - hours_forecast:]
    mean_value = train_river_data[[forecast_column]].median()[0]
    test_river_data["predicted_baseline"] = mean_value
    mse_baseline = sklearn.metrics.mean_squared_error(
        test_river_data[forecast_column], test_river_data["predicted_baseline"]
    )
    return test_river_data, round(mse_baseline, ndigits=3)


def plot_r2(river_flow_preds: pd.DataFrame) -> float:
    """
    We assume at this point river_flow_preds already has
    a predicted_baseline and a predicted_model column
    """
    pass


def get_model_r2_score(
        river_flow_df: pd.DataFrame,
        model_evaluate_function: Callable,
        forecast_column: str,
        hours_forecast=336,
):
    """

    model_evaluate_function should call any necessary preprocessing
    """
    test_river_data, baseline_mse = stream_baseline(
        river_flow_df, forecast_column)


def get_r2_value(model_mse, baseline_mse):
    return 1 - model_mse / baseline_mse


def get_value(the_path: str) -> None:
    df = pd.read_csv(the_path)
    res = stream_baseline(df, "cfs", 336)
    print(get_r2_value(0.120, res[1]))


def evaluate_model(model: Type[TimeSeriesModel],
                   model_type: str,
                   target_col: List[str],
                   evaluation_metrics: List,
                   dataset_params: {},
                   eval_log: Dict) -> Tuple[Dict, pd.DataFrame, int, pd.DataFrame]:
    """
    A function to evaluate a model. Called automatically at end of training.
    Can be imported for continuing to evaluate a model in other places as well.


    .. highlight:: python
    .. code-block:: python

        from hydroDL.evaluator import evaluate_model
        forecast_model = PyTorchForecast()
        evaluate_model(forecast_model, d, "cfs", )
        ...
    '''
    """
    if model_type == "PyTorch":
        # Firstly, test the trained model
        pred, obs, test_data = infer_on_torch_model(model, dataset_params)
        # To-do turn this into a general function
        probablistic = False
        print("Un-transforming data")
        if probablistic:
            # TODO: no probabilistic now
            print('probabilistic running on infer_on_torch_model')
        else:
            preds_np = test_data.inverse_scale(pred)
            obss_np = test_data.inverse_scale(obs)

    #  Then evaluate the model metrics
    crits = model.params["metrics"]
    # TODO: only support one output now
    inds = statError(obss_np[:, :, 0], preds_np[:, :, 0])
    for evaluation_metric in crits:
        eval_log[evaluation_metric] = inds[evaluation_metric]

    # Finally, try to explain model behaviour using shap
    # TODO: no SHAP now
    is_shap = False
    if probablistic:
        print("Probabilistic explainability currently not supported.")
    elif "n_targets" in model.params:
        print("Multitask forecasting support coming soon")
    elif is_shap:
        deep_explain_model_summary_plot(
            model, test_data, dataset_params["t_range_test"][0])
        deep_explain_model_heatmap(
            model, test_data, dataset_params["t_range_test"][0])

    return eval_log, preds_np, obss_np


def infer_on_torch_model(model: Type[TimeSeriesModel],
                         dataset_params: Dict = {},
                         ) -> (torch.Tensor, torch.Tensor, TestDataModel):
    """
    Function to handle both test evaluation and inference on a test data-frame.
    :return:
        pred: prediction
        obs: observation 
    :rtype: tuple()
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_model = TestDataModel(model.test_data)

    model.model.eval()
    all_data = test_model.load_test_data()
    pred = generate_predictions(model, test_model, *all_data[:-1], device=device, dataset_params=dataset_params)
    if pred.shape[1] != all_data[-1].shape[1]:
        # it means we use a Nto1 mode model, so we need to cutoff some previous data for observations to be comparable
        return pred, all_data[-1][:, test_model.test_data.rho - 1:, :], test_model
    return pred, all_data[-1], test_model


def generate_predictions(
        ts_model: Type[TimeSeriesModel],
        test_model: TestDataModel,
        *args,
        device: torch.device,
        dataset_params: {}
) -> torch.Tensor:
    model = ts_model.model
    if type(model) in sequence_first_model_lst:
        seq_first = True
    else:
        seq_first = False
    if issubclass(type(test_model.test_data), Dataset):
        test_loader = DataLoader(test_model.test_data, batch_size=dataset_params["batch_size"], shuffle=False)
        test_obs = []
        test_preds = []
        with torch.no_grad():
            for i_batch, (xs, ys) in enumerate(test_loader):
                if seq_first:
                    xs = xs.transpose(0, 1)
                    ys = ys.transpose(0, 1)
                xs = xs.to(device)
                ys = ys.to(device)
                output = model(xs)
                if seq_first:
                    xs = xs.transpose(0, 1)
                    ys = ys.transpose(0, 1)
                test_obs.append(ys.cpu().numpy())
                test_preds.append(output.cpu().numpy())
        pred = reduce(lambda x, y: np.vstack((x, y)), test_preds)
        if pred.shape[-1] == test_model.test_data.y.shape[-1]:
            # the len of 2nd dim is 1, which means we use an Nto1 mode
            pred = pred.flatten().reshape(test_model.test_data.y.shape[0], -1)
        pred = np.expand_dims(pred, axis=2)

    else:
        x = args[0]
        c = args[1]
        z = None
        if len(args) == 3:
            z = args[2]
        ngrid, nt, nx = x.shape
        if c is not None:
            nc = c.shape[-1]

        i_s = np.arange(0, ngrid, dataset_params["batch_size"])
        i_e = np.append(i_s[1:], ngrid)

        y_out_list = []
        for i in range(0, len(i_s)):
            print('batch {}'.format(i))
            x_temp = x[i_s[i]:i_e[i], :, :]

            if c is not None:
                c_temp = np.repeat(np.reshape(c[i_s[i]:i_e[i], :], [i_e[i] - i_s[i], 1, nc]), nt, axis=1)
                if seq_first:
                    xhTest = torch.from_numpy(np.swapaxes(np.concatenate([x_temp, c_temp], 2), 1, 0)).float()
                else:
                    xhTest = torch.from_numpy(np.concatenate([x_temp, c_temp], 2)).float()
            else:
                if seq_first:
                    xhTest = torch.from_numpy(np.swapaxes(x_temp, 1, 0)).float()
                else:
                    xhTest = torch.from_numpy(x_temp).float()
            xhTest = xhTest.to(device)
            if z is not None:
                # now only support z is 2d var
                assert z.ndim == 2
                if seq_first:
                    zTemp = z[i_s[i]:i_e[i], :]
                    zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
                else:
                    zTest = torch.from_numpy(z[i_s[i]:i_e[i], :]).float()
                zTest = zTest.to(device)
                y_p = model(xhTest, zTest)
            else:
                y_p = model(xhTest)
            if type(y_p) is tuple:
                others = y_p[1:]
                # Convention: y_p must be the first output of model
                y_p = y_p[0]
            if seq_first:
                y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
            else:
                y_out = y_p.detach().cpu().numpy()

            y_out_list.append(y_out)

        model.zero_grad()
        torch.cuda.empty_cache()
        data_stack = reduce(lambda a, b: np.vstack((a, b)),
                            list(map(lambda x: x.reshape(x.shape[0], x.shape[1]), y_out_list)))
        pred = np.expand_dims(data_stack, axis=2)

    return pred
