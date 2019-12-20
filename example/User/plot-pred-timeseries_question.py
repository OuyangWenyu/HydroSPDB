import os
import pandas as pd
import numpy as np
from hydroDL import stat
from visual import plot
from hydroDL import master

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

rootDB = os.path.join(cDir, 'input_VIC')
nEpoch = 100
out = os.path.join(cDir, 'output_VIC/CONUS_v8f1_SOILM_lev1_rho60_tr6')
tRange = [20160401, 20170401]

# load data
df, yp, yt = master.master_test(
    out, tRange=[20160401, 20170401], subset='CONUS_VICv8f1', epoch=100)
yp = yp.squeeze()
yt = yt.squeeze()


contents=pd.read_csv('/mnt/sdc/SUR_VIC/input_VIC/CONUS_VICv8f1/2016/PRES_FORA.csv',dtype=np.float,header=None).values

# print(contents)
# contents=contents-273.15


# calculate stat
statErr = stat.statError(yp, yt)
dataGrid = [statErr['RMSE'], statErr['Corr']]
# dataTs = [yp, yt]
dataTs = [contents]
t = df.getT()
crd = df.getGeo()
mapNameLst = ['RMSE', 'Correlation']
# tsNameLst = ['LSTM', 'VIC']
tsNameLst = ['PRES_FORA']
mapColor = None
tsColor = None

# print(statistics.mean(dataGrid[0]),statistics.mean(dataGrid[1]))

"""
fig1, ax1=plt.subplots()
ax1.set_title('RMSE', fontsize=18, fontweight='bold')
ax1.boxplot(dataGrid[0])
plt.savefig(out + '/RMSE.png')

fig2, ax2=plt.subplots()
ax2.set_title('CORR', fontsize=18, fontweight='bold')
ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.boxplot(dataGrid[1])
plt.savefig(out + '/COOR.png')
"""

# plot map and time series
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapColor=mapColor,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst)
