import hydroDL
import hydroDL.post.stat
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat
import numpy as np
from hydroDL import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import basemap

outFolder = r'/mnt/sdb/Data/Camels/test/'
df1 = camels.DataframeCsv(subset='all', tRange=[20050101, 20100101])
x1 = df1.getDataTS(varLst=camels.forcingLst, doNorm=True, rmNan=True)
y1 = df1.get_data_obs(doNorm=True, rmNan=False)
c1 = df1.get_data_const(varLst=camels.attrLstSel, doNorm=True, rmNan=True)
yt1 = df1.get_data_obs(doNorm=False, rmNan=False).squeeze()

dfz1 = camels.DataframeCsv(subset='all', tRange=[20141231, 20091231])
z1 = dfz1.get_data_obs(doNorm=True, rmNan=False)

dfz2 = camels.DataframeCsv(subset='all', tRange=[20141227, 20091227])
z2 = dfz2.get_data_obs(doNorm=True, rmNan=False)

df2 = camels.DataframeCsv(subset='all', tRange=[20100101, 20150101])
x2 = df2.getDataTS(varLst=camels.forcingLst, doNorm=True, rmNan=True)
c2 = df2.get_data_const(varLst=camels.attrLstSel, doNorm=True, rmNan=True)
yt2 = df2.get_data_obs(doNorm=False, rmNan=False).squeeze()

model = train.load_model(outFolder, 100, modelName='test')
yp1 = train.test_model(model, x1, c1)
yp1 = hydroDL.post.stat.trans_norm(yp1, 'usgsFlow', toNorm=False).squeeze()
yp2 = train.test_model(model, x2, c2)
yp2 = hydroDL.post.stat.trans_norm(yp2, 'usgsFlow', toNorm=False).squeeze()

statErr1 = stat.statError(yp1, yt2)
statErr2 = stat.statError(yp2, yt2)
dataMap = [statErr2['Corr'], statErr1['Corr'] - statErr2['Corr']]
dataTs = [yt2, yp2]
t = df2.getT()
crd = df2.getGeo()
mapNameLst = ['Test Corr', 'Train Corr - Test Corr']
tsNameLst = ['USGS','LSTM']
colorMap = None
colorTs = None

import imp
imp.reload(plot)
plot.plotTsMap(
    dataMap,
    dataTs,
    lat=crd[:, 0],
    lon=crd[:, 1],
    t=t,
    colorMap=colorMap,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst)

# plot time series
t = utils.time.tRange2Array([20100101, 20150101])
fig, axes = plt.subplots(5, 1, figsize=(12, 8))
iLst = [10, 52, 178, 404, 620]
for k in range(5):
    iGrid = np.random.randint(0, 671)
    iGrid = iLst[k]
    yPlot = [yt2[iGrid, :], yp2[iGrid, :]]
    gageId = camels.gageDict['id'][iGrid]
    if k == 0:
        plot.plotTS(
            t,
            yPlot,
            ax=axes[k],
            cLst='br',
            markerLst='--',
            legLst=['USGS', 'LSTM'])

    else:
        plot.plotTS(t, yPlot, ax=axes[k], cLst='br', markerLst='--')
    axes[k].set_ylabel('streamflow(cfs)')
    if k == 4:
        axes[k].set_xlabel('time')
plt.tight_layout()
fig.show()
