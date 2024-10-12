# -*- coding: utf-8 -*-
import os.path
import shelve
import math
import numpy as np
from sklearn.metrics import mean_squared_error
def to2dNpArray(x):
    # type conversion
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    # convert to matrix
    if x.ndim == 1:
        x = x[np.newaxis, :]
    return x

def to2dColVec(x):
    """convert to column vector
    convert a number or 1-d vector to column vector
    """
    if np.size(x) == 1:
        return x
    # convert to 2-d column vector of type numpy.array
    return np.reshape(x, (np.size(x), -1))

def get_root():
    root = os.path.dirname(__file__)
    return root

def get_sort(pop, surro, mse):
    mse_sum = np.sum(mse)
    mse_len = len(mse)
    w = []
    for m in mse:
        _a = mse_sum - m
        _b = (mse_len-1) * mse_sum
        w.append(_a/_b)
    pl = np.zeros((len(pop), mse_len))
    wpl = np.zeros((len(pop), mse_len))
    for j in range(mse_len):
        temp = surro[j].predict(pop)
        pl[:, j] = temp
        wpl[:, j] = temp * w[j]
    row_std = np.std(pl, axis=1)
    row_mean = np.sum(wpl, axis=1)
    return row_mean, row_std

def get_pseudo(pop, surro, mse):
    row_mean, row_std = get_sort(pop, surro, mse)
    num = np.argsort(row_std)
    top = num[:3]
    new_x = pop[top].copy()
    new_y = row_mean[top].copy()
    return new_x, new_y

def get_wsuro(pop, surros, rmses):
    row_mean, row_std = get_sort(pop, surros, rmses)
    return row_mean

def get_elit(pop, surro, mse):

    row_mean, row_std = get_sort(pop, surro, mse)
    num = np.argmin(row_mean)
    return pop[num], row_mean[num]












