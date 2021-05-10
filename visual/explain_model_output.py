import random
from datetime import datetime
from typing import Optional
import numpy as np
import shap
import torch

from data.data_loaders import TestDataModel
from visual.plot_functions import (
    plot_shap_value_heatmaps,
    plot_shap_values_from_history,
    plot_summary_shap_values,
    plot_summary_shap_values_over_time_series,
)

BACKGROUND_BATCH_SIZE = 5


def _prepare_background_tensor(csv_test_loader: TestDataModel, backgound_batch_size: int = BACKGROUND_BATCH_SIZE
                               ) -> torch.Tensor:
    """Generate background batches for deep explainer.
    Random sample batches as background data
    background tensor of size (batch_size, history_len, num_feature)
    Args:
        csv_test_loader (CSVTestLoader): test data loader
        backgound_batch_size (int): number of batches used as background data
        for deep explainer. Default to BACKGROUND_BATCH_SIZE.
    Returns:
        torch.Tensor: background tensor of size
        (batch_size, history_len, num_feature)
    """
    background_data = csv_test_loader.original_df
    background_batches = csv_test_loader.convert_history_batches(csv_test_loader.df.columns, background_data)
    # remove last batch in the list because it may not be of
    # size (history_len, num_feature) due to length of original df
    background_tensor = torch.stack(random.sample(background_batches[:-1], backgound_batch_size)).float()
    return background_tensor


def deep_explain_model_summary_plot(model, csv_test_loader: TestDataModel, datetime_start: Optional[datetime] = None
                                    ) -> None:
    """Generate feature summary plot for trained deep learning models
    Args:
        model (object): trained model
        csv_test_loader (CSVTestLoader): test data loader
        datetime_start (datetime, optional): start date of the test prediction,
            Defaults to None, i.e. using model inference parameters.
    """
    if model.params["model_params"]["model_name"] == "SimpleTransformer":
        print("SimpleTransformer currently not supported.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model.params["model_params"]["model_name"] == "DARNN" and device.type == "cuda":
        print("DARNN does not work with shap on CUDA")
        return

    history, _, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)
    background_tensor = _prepare_background_tensor(csv_test_loader)
    background_tensor = background_tensor.to(device)
    model.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    deep_explainer = shap.DeepExplainer(model.model, background_tensor)
    shap_values = deep_explainer.shap_values(background_tensor)
    shap_values = np.stack(shap_values)
    # shap_values needs to be 4-dimensional
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(shap_values, names=["preds", "batches", "observations", "features"])

    # summary plot shows overall feature ranking
    # by average absolute shap values
    fig = plot_summary_shap_values(shap_values, csv_test_loader.df.columns)
    abs_mean_shap_values = shap_values.mean(axis=["preds", "batches"])
    multi_shap_values = abs_mean_shap_values.mean(axis="observations")
    # summary plot for multi-step outputs
    # multi_shap_values = shap_values.apply_along_axis(np.mean, 'batches')
    fig = plot_summary_shap_values_over_time_series(
        shap_values, csv_test_loader.df.columns
    )

    # summary plot for one prediction at datetime_start

    history = history.to(device).unsqueeze(0)
    history_numpy = torch.tensor(history.cpu().numpy(), names=["batches", "observations", "features"])

    shap_values = deep_explainer.shap_values(history)
    shap_values = np.stack(shap_values)
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(shap_values, names=["preds", "batches", "observations", "features"])

    figs = plot_shap_values_from_history(shap_values, history_numpy)


def deep_explain_model_heatmap(model, csv_test_loader: TestDataModel, datetime_start: Optional[datetime] = None
                               ) -> None:
    """Generate feature heatmap for prediction at a start time
    Args:
        model ([type]): trained model
        csv_test_loader ([CSVTestLoader]): test data loader
        datetime_start (Optional[datetime], optional): start date of the test prediction,
            Defaults to None, i.e. using model inference parameters.
    Returns:
        None
    """
    if model.params["model_name"] == "SimpleTransformer":
        print("SimpleTransformer currently not supported.")
        return
    elif "probabilistic" in model.params:
        print("Probabilistic currently not supported.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model.params["model_name"] == "DARNN" and device.type == "cuda":
        print("Currently DARNN doesn't work with shap on CUDA")
        return

    if datetime_start is None:
        datetime_start = model.params["inference_params"]["datetime_start"]

    history, _, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)
    background_tensor = _prepare_background_tensor(csv_test_loader)
    background_tensor = background_tensor.to(device)
    model.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    # for each element in each N x M batch in L,
    # attribute to each prediction in forecast len
    deep_explainer = shap.DeepExplainer(model.model, background_tensor)
    shap_values = deep_explainer.shap_values(background_tensor)  # forecast_len x N x L x M
    shap_values = np.stack(shap_values)
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(shap_values, names=["preds", "batches", "observations", "features"])
    figs = plot_shap_value_heatmaps(shap_values)

    # heatmap one prediction sequence at datetime_start
    # (seq_len*forecast_len) per fop feature
    to_explain = history.to(device).unsqueeze(0)
    shap_values = deep_explainer.shap_values(to_explain)
    shap_values = np.stack(shap_values)
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(shap_values, names=["preds", "batches", "observations", "features"])

    figs = plot_shap_value_heatmaps(shap_values)
