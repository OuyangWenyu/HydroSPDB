import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# train_process = pd.read_csv(
#     "/home/owen/Documents/Code/hydro-anthropogenic-lstm/example/data/gages/rnnStreamflow/All-90-95/run.csv",
#     header=None)
train_process = pd.read_csv(
    "/example/data/camels/rnnStreamflow/All-90-95/run.csv",
    header=None)
train_process_ = train_process.iloc[:, 0].str.split()
p_x = np.array([])
p_y = np.array([])
for process in train_process_:
    p_x = np.append(p_x, int(process[1]))
    p_y = np.append(p_y, float(process[3]))
plt.plot(p_x, p_y)
plt.show()
