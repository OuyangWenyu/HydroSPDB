import numpy as np


def interpNan(x, mode='linear'):
    if len(x.shape) == 1:
        ngrid = 1
        nt = x.shape[0]
    else:
        ngrid, nt = x.shape
    for k in range(ngrid):
        xx = x[k, :]
        xx = interpNan1d(xx, mode)
    return x


def interpNan1d(x, mode='linear'):
    i0 = np.where(np.isnan(x))[0]
    i1 = np.where(~np.isnan(x))[0]
    if len(i1) > 0:
        if mode is 'linear':
            x[i0] = np.interp(i0, i1, x[i1])
        if mode is 'pre':
            x0 = x[i1[0]]
            for k in range(len(x)):
                if k in i0:
                    x[k] = x0
                else:
                    x0 = x[k]

    return x


def concat_two_3darray(arr1, arr2):
    arr3 = np.zeros([arr1.shape[0], arr1.shape[1], arr1.shape[2] + arr2.shape[2]])
    for j in range(arr1.shape[0]):
        arr3[j] = np.concatenate((arr1[j], arr2[j]), axis=1)
    return arr3
