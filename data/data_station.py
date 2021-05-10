from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import os
import json
from typing import Set, Dict, List
import requests
from datetime import datetime, timedelta

from hydroDL.da_rnn.custom_types import TrainData


def get_closest_gage(
        gage_df: pd.DataFrame,
        station_df: pd.DataFrame,
        path_dir: str,
        start_row: int,
        end_row: int):
    # Function that calculates the closest weather stations to gage and stores in JSON
    # Base u
    for row in range(start_row, end_row):
        gage_info = {}
        gage_info["river_id"] = int(gage_df.iloc[row]['id'])
        gage_lat = gage_df.iloc[row]['latitude']
        gage_long = gage_df.iloc[row]['logitude']
        gage_info["stations"] = []
        total = len(station_df.index)
        for i in range(0, total):
            stat_row = station_df.iloc[i]
            dist = haversine(stat_row["lon"], stat_row["lat"], gage_long, gage_lat)
            st_id = stat_row['stid']
            gage_info["stations"].append({"station_id": st_id, "dist": dist})
        # This bug was actually only later discovered that it puts further away stations first.
        # However subsequent code was then based on it. So we just use negative
        # indices later on (i.e [-20:])
        gage_info["stations"] = sorted(gage_info['stations'], key=lambda i: i["dist"], reverse=True)
        with open(os.path.join(path_dir, str(gage_info["river_id"]) + "stations.json"), 'w') as w:
            count = 0
            json.dump(gage_info, w)
            if count % 100 == 0:
                print("Currently at " + str(count))
            count += 1


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_weather_data(file_path: str, econet_gages: Set, base_url: str):
    """
    Function that retrieves if station has weather
    data for a specific gage either from ASOS or ECONet
    """
    # Base URL
    # "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={}&data=tmpf&data=p01m&year1=2019&month1=1&day1=1&year2=2019&month2=1&day2=2&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"

    gage_meta_info = {}

    with open(file_path) as f:
        gage_data = json.load(f)
    gage_meta_info["gage_id"] = gage_data["river_id"]
    gage_meta_info["stations"] = []
    closest_stations = gage_data["stations"][-20:]
    for station in reversed(closest_stations):
        url = base_url.format(station["station_id"])
        response = requests.get(url)
        if len(response.text) > 100:
            print(response.text)
            gage_meta_info["stations"].append({"station_id": station["station_id"],
                                               "dist": station["dist"], "cat": "ASOS"})
        elif station["station_id"] in econet_gages:
            gage_meta_info["stations"].append({"station_id": station["station_id"],
                                               "dist": station["dist"], "cat": "ECO"})
    return gage_meta_info


def format_dt(date_time_str: str) -> datetime:
    proper_datetime = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
    if proper_datetime.minute != 0:
        proper_datetime = proper_datetime + timedelta(hours=1)
        proper_datetime = proper_datetime.replace(minute=0)
    return proper_datetime


def convert_temp(temparature: str) -> float:
    """
    Note here temp could be a number or 'M'
    which stands for missing. We use 50 at the moment
    to fill missing values.
    """
    try:
        return float(temparature)
    except BaseException:
        return 50


def process_asos_data(file_path: str, base_url: str) -> Dict:
    """
    Function that saves the ASOS data to CSV
    uses output of get weather data.
    """
    with open(file_path) as f:
        gage_data = json.load(f)
        for station in gage_data["stations"]:
            if station["cat"] == "ASOS":
                response = requests.get(base_url.format(station["station_id"]))
                with open("temp_weather_data.csv", "w+") as f:
                    f.write(response.text)
                df, missing_precip, missing_temp = process_asos_csv("temp_weather_data.csv")
                station["missing_precip"] = missing_precip
                station["missing_temp"] = missing_temp
                df.to_csv(str(gage_data["gage_id"]) + "_" + str(station["station_id"]) + ".csv")
    with open(file_path, "w") as f:
        json.dump(gage_data, f)
    return gage_data


def process_asos_csv(path: str):
    df = pd.read_csv(path)
    missing_precip = df['p01m'][df['p01m'] == 'M'].count()
    missing_temp = df['tmpf'][df['tmpf'] == 'M'].count()
    df['hour_updated'] = df['valid'].map(format_dt)
    df['tmpf'] = pd.to_numeric(df['tmpf'], errors='coerce')

    df['p01m'] = pd.to_numeric(df['p01m'], errors='coerce')
    # Originally the idea for imputation was to
    # replace missing values with an average of the two closest values
    # But since ASOS stations record at different intervals this could
    # actually cause an overestimation of precip. Instead for now we are replacing with 0
    # df['p01m']=(df['p01m'].fillna(method='ffill') + df['p01m'].fillna(method='bfill'))/2
    df['p01m'] = df['p01m'].fillna(0)
    df['tmpf'] = (df['tmpf'].fillna(method='ffill') + df['tmpf'].fillna(method='bfill')) / 2
    df = df.groupby(by=['hour_updated'], as_index=False).agg(
        {'p01m': 'sum', 'valid': 'first', 'tmpf': 'mean'})
    return df, int(missing_precip), int(missing_temp)


def make_temporal_features(features_list: Dict, dt_column: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create features
    """
    df[dt_column] = df[dt_column].to_datetime()
    for key, value in features_list.items():
        df[key] = df[dt_column].map(value)
    return df


def create_feature(key: str, value: str, df: pd.DataFrame, dt_column: str):
    """Function to create temporal features

    :param key: The datetime feature you would like to create
    :type key: str
    :param value: The type of feature you would like to create (cyclical or numerical)
    :type value: str
    :param df: The Pandas dataframe with the datetime
    :type df: pd.DataFrame
    :param dt_column: The name of the datetime column
    :type dt_column: str
    :return: The dataframe with the newly added column
    :rtype: pd.DataFrame w
    """
    if key == "day_of_week":
        df[key] = df[dt_column].map(lambda x: x.weekday())
    elif key == "hour":
        df[key] = df[dt_column].map(lambda x: x.hour)
    elif key == "day":
        df[key] = df[dt_column].map(lambda x: x.day)
    elif key == "month":
        df[key] = df[dt_column].map(lambda x: x.month)
    elif key == "year":
        df[key] = df[dt_column].map(lambda x: x.year)
    if value == "cyclical":
        df = cyclical(df, key)
    return df


def feature_fix(preprocess_params: Dict, dt_column: str, df: pd.DataFrame):
    """Adds temporal features

    :param preprocess_params: Dictionary of temporal parameters e.g. {"day":"numerical"}
    :type preprocess_params: Dict
    :param dt_column: The column name of the data
    :param df: The dataframe to add the temporal features to
    :type df: pd.DataFrame
    :return: Returns the new data-frame and a list of the new column names
    :rtype: Tuple(pd.Dataframe, List[str])

    .. code-block:: python
        feats_to_add = {"month":"cyclical", "day":"numerical"}
        df, column_names feature_fix(feats_to_add, "datetime")
        print(column_names) # ["cos_month", "sin_month", "day"]
    """
    print("Running code to add temporal features")
    column_names = []
    if "datetime_params" in preprocess_params:
        for key, value in preprocess_params["datetime_params"].items():
            df = create_feature(key, value, df, dt_column)
            if value == "cyclical":
                column_names.append("cos_" + key)
                column_names.append("sin_" + key)
            else:
                column_names.append(key)
    return df, column_names


def cyclical(df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
    """ A function to create cyclical encodings for Pandas data-frames.

    :param df: A Pandas Dataframe where you want the dt encoded
    :type df: pd.DataFrame
    :param feature_column: The name of the feature column. Should be either (day_of_week, hour, month, year)
    :type feature_column: str
    :return: The dataframew with three new columns: norm_feature, cos_feature, sin_feature
    :rtype: pd.DataFrame
    """
    df["norm"] = 2 * np.pi * df[feature_column] / df[feature_column].max()
    df['cos_' + feature_column] = np.cos(df['norm'])
    df['sin_' + feature_column] = np.sin(df['norm'])
    return df


def format_data(dat, targ_column: List[str]) -> TrainData:
    # Test numpy conversion
    proc_dat = dat.to_numpy()
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in targ_column:
        mask[dat_cols.index(col_name)] = False
    feats = proc_dat[:, mask].astype(float)
    targs = proc_dat[:, ~mask].astype(float)
    return TrainData(feats, targs)


def make_data(
    csv_path: str,
    target_col: List[str],
    test_length: int,
    relevant_cols=[
        "cfs",
        "temp",
        "precip"]) -> TrainData:
    """
    Returns full preprocessed data.
    Does not split train/test that must be done later.
    """
    final_df = pd.read_csv(csv_path)
    print(final_df.shape[0])
    if len(target_col) > 1:
        # Restrict target columns to height and cfs. Alternatively could replace this with loop
        height_df = final_df[[target_col[0], target_col[1], 'precip', 'temp']]
        height_df.columns = [target_col[0], target_col[1], 'precip', 'temp']
    else:
        height_df = final_df[[target_col[0]] + relevant_cols]
    preprocessed_data2 = format_data(height_df, target_col)
    return preprocessed_data2


def fix_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic function to fix initil data bug
    related to NaN values in non-eastern-time zones due
    to UTC conversion.
    """
    the_count = df[0:2]['cfs'].isna().sum()
    return df[the_count:]


def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to fill missing values with nearest value.
    Should be run only after splitting on the NaN chunks.
    """
    df = fix_timezones(df)
    df['cfs1'] = df['cfs'].interpolate(method='nearest').ffill().bfill()
    df['precip'] = df['p01m'].interpolate(method='nearest').ffill().bfill()
    df['temp'] = df['tmpf'].interpolate(method='nearest').ffill().bfill()
    return df


def forward_back_generic(df: pd.DataFrame, relevant_columns: List) -> pd.DataFrame:
    """
    Function to fill missing values with nearest value (forward first)
    """
    for col in relevant_columns:
        df[col] = df[col].interpolate(method='nearest').ffill().bfill()
    return df


def back_forward_generic(df: pd.DataFrame, relevant_columns: List) -> pd.DataFrame:
    """
    Function to fill missing values with nearest values (backward first)
    """
    for col in relevant_columns:
        df[col] = df[col].interpolate(method='nearest').bfill().ffill()
    return df