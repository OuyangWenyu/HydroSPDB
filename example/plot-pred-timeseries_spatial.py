import os
import statistics
import matplotlib.pyplot as plt
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat
from hydroDL import master

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

rootDB = os.path.join(cDir, 'input_VIC')
nEpoch = 100
out = os.path.join(cDir, 'output_VIC/CONUS_v8f1_SOILM_lev1_rho60_pix50_ep800_tr6_all_const')
tRange = [20150401, 20160401]

# load data
df, yp, yt = master.test(
    out, tRange=[20150401, 20160401], subset='CONUS_VICv8f2', epoch=100)
yp = yp.squeeze()
yt = yt.squeeze()

# calculate stat
statErr = stat.statError(yp, yt)
dataGrid = [statErr['RMSE'], statErr['Corr']]
dataTs = [yp, yt]
# dataTs = [yp]
t = df.getT()
crd = df.getGeo()
mapNameLst = ['RMSE', 'Correlation']
tsNameLst = ['LSTM', 'VIC']
# tsNameLst = ['VIC']
mapColor = None
tsColor = None

# print(statistics.mean(dataGrid[0]),statistics.mean(dataGrid[1]))

fig1, ax1=plt.subplots()
ax1.set_title('RMSE', fontsize=18, fontweight='bold')
ax1.boxplot(dataGrid[0])
plt.savefig(out + '/RMSE.png')

fig2, ax2=plt.subplots()
ax2.set_title('CORR', fontsize=18, fontweight='bold')
ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.boxplot(dataGrid[1])
plt.savefig(out + '/COOR.png')

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
