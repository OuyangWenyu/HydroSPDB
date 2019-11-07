# read camels dataset
import os
import pandas as pd
import numpy as np
from hydroDL import utils, pathCamels
from pandas.api.types import is_numeric_dtype, is_string_dtype
import time
import json

from hydroDL.post.stat import cal_stat, trans_norm
from . import Dataframe

# module variable
dirDB = pathCamels['DB']
gageFldLst = ['huc', 'id', 'name', 'lat', 'lon', 'area']
tRange = [19800101, 20150101]
tLst = utils.time.tRange2Array(tRange)
nt = len(tLst)
forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
attrLstSel = [
    'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
    'lai_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
    'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    'max_water_content', 'geol_1st_class', 'geol_2nd_class', 'geol_porostiy',
    'geol_permeability'
]


def read_gage_info(dir_db):
    gage_file = os.path.join(dir_db, 'basin_timeseries_v1p2_metForcing_obsFlow',
                             'basin_dataset_public_v1p2', 'basin_metadata',
                             'gauge_information.txt')
    # 指定第2列读取为字符串类型，否则作为数值类型，会自动把编号最前面的0省去，和实际编号不符
    data = pd.read_csv(gage_file, sep='\t', header=None, skiprows=1, dtype={1: str})
    # header gives some troubles. Skip and hardcode

    out = dict()
    for s in gageFldLst:
        if s is 'name':
            out[s] = data[gageFldLst.index(s)].values.tolist()
        else:
            out[s] = data[gageFldLst.index(s)].values
    return out


# module variable
gageDict = read_gage_info(dirDB)


def readUsgsGage(usgsId, *, readQc=False):
    ind = np.argwhere(gageDict['id'] == usgsId)[0][0]
    huc = gageDict['huc'][ind]
    if type(usgsId) is str:
        usgsId = int(usgsId)
    usgsFile = os.path.join(dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
                            'basin_dataset_public_v1p2', 'usgs_streamflow',
                            str(huc).zfill(2),
                            '%08d_streamflow_qc.txt' % usgsId)
    dataTemp = pd.read_csv(usgsFile, sep=r'\s+', header=None)
    obs = dataTemp[4].values
    obs[obs < 0] = np.nan
    if readQc is True:
        qcDict = {'A': 1, 'A:e': 2, 'M': 3}
        qc = np.array([qcDict[x] for x in dataTemp[5]])
    if len(obs) != nt:
        out = np.full([nt], np.nan)
        dfDate = dataTemp[[1, 2, 3]]
        dfDate.columns = ['year', 'month', 'day']
        date = pd.to_datetime(dfDate).values.astype('datetime64[D]')
        [C, ind1, ind2] = np.intersect1d(date, tLst, return_indices=True)
        out[ind2] = obs
        if readQc is True:
            outQc = np.full([nt], np.nan)
            outQc[ind2] = qc
    else:
        out = obs
        if readQc is True:
            outQc = qc

    if readQc is True:
        return out, outQc
    else:
        return out


def readUsgs(usgsIdLst):
    t0 = time.time()
    y = np.empty([len(usgsIdLst), nt])
    for k in range(len(usgsIdLst)):
        dataObs = readUsgsGage(usgsIdLst[k])
        y[k, :] = dataObs
    print("read usgs streamflow", time.time() - t0)
    return y


def read_forcing_gage(usgs_id, var_lst=forcingLst, *, dataset='nldas'):
    # dataset = daymet or maurer or nldas
    forcing_lst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    ind = np.argwhere(gageDict['id'] == usgs_id)[0][0]
    huc = gageDict['huc'][ind]

    data_folder = os.path.join(
        dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
        'basin_dataset_public_v1p2', 'basin_mean_forcing')
    if dataset is 'daymet':
        temp_s = 'cida'
    else:
        temp_s = dataset
    if type(usgs_id) is str:
        usgs_id = int(usgs_id)
    data_file = os.path.join(data_folder, dataset,
                             str(huc).zfill(2),
                             '%08d_lump_%s_forcing_leap.txt' % (usgs_id, temp_s))
    data_temp = pd.read_csv(data_file, sep=r'\s+', header=None, skiprows=4)
    nf = len(var_lst)
    out = np.empty([nt, nf])
    for k in range(nf):
        # assume all files are of same columns. May check later.
        ind = forcing_lst.index(var_lst[k])
        out[:, k] = data_temp[ind + 4].values
    return out


def read_forcing(usgs_id_lst, var_lst):
    """读取camels文件夹下的nldas文件夹下的驱动数据"""
    t0 = time.time()
    x = np.empty([len(usgs_id_lst), nt, len(var_lst)])
    for k in range(len(usgs_id_lst)):
        data = read_forcing_gage(usgs_id_lst[k], var_lst)
        x[k, :, :] = data
    print("read usgs streamflow", time.time() - t0)
    return x


def readAttrAll(*, saveDict=False):
    dataFolder = os.path.join(dirDB, 'camels_attributes_v2.0',
                              'camels_attributes_v2.0')
    fDict = dict()  # factorize dict
    varDict = dict()
    varLst = list()
    outLst = list()
    keyLst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']

    for key in keyLst:
        dataFile = os.path.join(dataFolder, 'camels_' + key + '.txt')
        dataTemp = pd.read_csv(dataFile, sep=';')
        varLstTemp = list(dataTemp.columns[1:])
        varDict[key] = varLstTemp
        varLst.extend(varLstTemp)
        k = 0
        nGage = len(gageDict['id'])
        outTemp = np.full([nGage, len(varLstTemp)], np.nan)
        for field in varLstTemp:
            if is_string_dtype(dataTemp[field]):
                value, ref = pd.factorize(dataTemp[field], sort=True)
                outTemp[:, k] = value
                fDict[field] = ref.tolist()
            elif is_numeric_dtype(dataTemp[field]):
                outTemp[:, k] = dataTemp[field].values
            k = k + 1
        outLst.append(outTemp)
    out = np.concatenate(outLst, 1)
    if saveDict is True:
        fileName = os.path.join(dataFolder, 'dictFactorize.json')
        with open(fileName, 'w') as fp:
            json.dump(fDict, fp, indent=4)
        fileName = os.path.join(dataFolder, 'dictAttribute.json')
        with open(fileName, 'w') as fp:
            json.dump(varDict, fp, indent=4)
    return out, varLst


def readAttr(usgsIdLst, varLst):
    attrAll, varLstAll = readAttrAll()
    indVar = list()
    for var in varLst:
        indVar.append(varLstAll.index(var))
    idLstAll = gageDict['id']
    C, indGrid, ind2 = np.intersect1d(idLstAll, usgsIdLst, return_indices=True)
    temp = attrAll[indGrid, :]
    out = temp[:, indVar]
    return out


def calStatAll():
    statDict = dict()
    idLst = gageDict['id']
    # usgs streamflow
    y = readUsgs(idLst)
    statDict['usgsFlow'] = cal_stat(y)
    # forcing
    x = read_forcing(idLst, forcingLst)
    for k in range(len(forcingLst)):
        var = forcingLst[k]
        statDict[var] = cal_stat(x[:, :, k])
    # const attribute
    attrData, attrLst = readAttrAll()
    for k in range(len(attrLst)):
        var = attrLst[k]
        statDict[var] = cal_stat(attrData[:, k])
    statFile = os.path.join(dirDB, 'Statistics.json')
    with open(statFile, 'w') as fp:
        json.dump(statDict, fp, indent=4)


# module variable
statFile = os.path.join(dirDB, 'Statistics.json')
if not os.path.isfile(statFile):
    calStatAll()
with open(statFile, 'r') as fp:
    statDict = json.load(fp)


def createSubsetAll(opt, **kw):
    if opt is 'all':
        idLst = gageDict['id']
        subsetFile = os.path.join(dirDB, 'Subset', 'all.csv')
        np.savetxt(subsetFile, idLst, delimiter=',', fmt='%d')


class DataframeCamels(Dataframe):
    def __init__(self, *, subset='All', t_range):
        self.rootDB = dirDB
        self.subset = subset
        if subset == 'All':  # change to read subset later
            self.usgsId = gageDict['id']
            crd = np.zeros([len(self.usgsId), 2])
            crd[:, 0] = gageDict['lat']
            crd[:, 1] = gageDict['lon']
            self.crd = crd
        self.time = utils.time.tRange2Array(t_range)

    def getGeo(self):
        return self.crd

    def getT(self):
        return self.time

    def get_data_obs(self, *, do_norm=True, rm_nan=True):
        data = readUsgs(self.usgsId)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        if do_norm is True:
            data = trans_norm(data, 'usgsFlow', statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, *, var_lst=forcingLst, do_norm=True, rm_nan=True):
        if type(var_lst) is str:
            var_lst = [var_lst]
        # read ts forcing
        data = read_forcing(self.usgsId, var_lst)
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        if do_norm is True:
            data = trans_norm(data, var_lst, statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, *, var_lst=attrLstSel, do_norm=True, rm_nan=True):
        if type(var_lst) is str:
            var_lst = [var_lst]
        data = readAttr(self.usgsId, var_lst)
        if do_norm is True:
            data = trans_norm(data, var_lst, statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data
