import numpy
import numpy as np
import torch


def flat_data(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return xSort


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


def is_any_elem_in_a_lst(lst1, lst2, return_index=False, include=False):
    do_exist = False
    idx_lst = []
    for j in range(len(lst1)):
        if include:
            for lst2_elem in lst2:
                if lst1[j] in lst2_elem:
                    idx_lst.append(j)
                    do_exist = True
        else:
            if lst1[j] in lst2:
                idx_lst.append(j)
                do_exist = True
    if return_index:
        return do_exist, idx_lst
    return do_exist


def random_choice_no_return(arr, num_lst):
    """sampling without replacement multi-times, and the num of each time is in num_lst"""
    num_lst_arr = np.array(num_lst)
    num_sum = num_lst_arr.sum()
    if type(arr) == list:
        arr = np.array(arr)
    assert num_sum <= arr.size
    results = []
    arr_residue = np.arange(arr.size)
    for num in num_lst_arr:
        idx_chosen = np.random.choice(arr_residue.size, num, replace=False)
        chosen_idx_in_arr = np.sort(arr_residue[idx_chosen])
        results.append(arr[chosen_idx_in_arr])
        arr_residue = np.delete(arr_residue, idx_chosen)
    return results


def find_integer_factors_close_to_square_root(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


def concat_two_3darray(arr1, arr2):
    arr3 = np.zeros([arr1.shape[0], arr1.shape[1], arr1.shape[2] + arr2.shape[2]])
    for j in range(arr1.shape[0]):
        arr3[j] = np.concatenate((arr1[j], arr2[j]), axis=1)
    return arr3


def copy_attr_array_in2d(arr1, len_of_2d):
    arr2 = np.zeros([arr1.shape[0], len_of_2d, arr1.shape[1]])
    for k in range(arr1.shape[0]):
        arr2[k] = np.tile(arr1[k], arr2.shape[1]).reshape(arr2.shape[1], arr1.shape[1])
    return arr2


def random_index(ngrid, nt, dim_subset):
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, [batch_size])
    i_t = np.random.randint(0, nt - rho, [batch_size])
    return i_grid, i_t


def select_subset(x, i_grid, i_t, rho, *, c=None, tuple_out=False):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(i_grid):  # hack
        i_grid = np.arange(0, len(i_grid))  # hack
        if nt <= rho:
            i_t.fill(0)
    if i_t is not None:
        batch_size = i_grid.shape[0]
        x_tensor = torch.zeros([rho, batch_size, nx], requires_grad=False)
        for k in range(batch_size):
            temp = x[i_grid[k]:i_grid[k] + 1, np.arange(i_t[k], i_t[k] + rho), :]
            x_tensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if len(x.shape) == 2:
            x_tensor = torch.from_numpy(x[i_grid, :]).float()
        else:
            x_tensor = torch.from_numpy(np.swapaxes(x[i_grid, :, :], 1, 0)).float()
            rho = x_tensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho, axis=1)
        c_tensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()
        if tuple_out:
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                c_tensor = c_tensor.cuda()
            out = (x_tensor, c_tensor)
        else:
            out = torch.cat((x_tensor, c_tensor), 2)
    else:
        out = x_tensor
    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out


def select_subset_batch_first(x, i_grid, i_t, rho, *, c=None):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] < len(i_grid):
        raise ValueError('grid num should be smaller than x.shape[0]')
    if nt < rho:
        raise ValueError('time length option should be larger than rho')

    batch_size = i_grid.shape[0]
    x_tensor = torch.zeros([batch_size, rho, nx], requires_grad=False)
    for k in range(batch_size):
        x_tensor[k:k + 1, :, :] = torch.from_numpy(
            x[i_grid[k]:i_grid[k] + 1, np.arange(i_t[k], i_t[k] + rho), :]).float()

    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho, axis=1)
        c_tensor = torch.from_numpy(temp).float()
        out = torch.cat((x_tensor, c_tensor), 2)
    else:
        out = x_tensor
    if torch.cuda.is_available():
        out = out.cuda()
    return out


def select_subset_seq(x, i_grid, i_t, rho, *, c=None, seq_len=100):
    assert len(x.shape) == 2
    if x.shape[0] < len(i_grid):
        raise ValueError('grid num should be smaller than x.shape[0]')
    if x.shape[1] < rho or x.shape[1] < seq_len:
        raise ValueError('time length option should be larger than rho and seq_len')
    batch_size = i_grid.shape[0]
    x_tensor = torch.zeros([rho, batch_size, seq_len], requires_grad=False)
    for k in range(batch_size):
        x_temp = x[i_grid[k]:i_grid[k] + 1, np.arange(i_t[k], i_t[k] + seq_len - 1 + rho)]
        temp = np.zeros([rho, 1, seq_len])
        for i in range(temp.shape[0]):
            temp[i, :, :] = x_temp[:, i:i + seq_len]
        x_tensor[:, k:k + 1, :] = torch.from_numpy(temp)

    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho, axis=1)
        c_tensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()
        out = torch.cat((x_tensor, c_tensor), 2)
    else:
        out = x_tensor
    if torch.cuda.is_available():
        out = out.cuda()
    return out
