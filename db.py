import copy
import os.path
import shelve
import numpy as np
import sampling
from rbfn import RBFN
from sklearn.metrics import mean_squared_error, mean_absolute_error
from copy import deepcopy

import math
class DB(object):
    def __init__(self, kw):
        self.d = kw["dim"]
        self.low = kw["lower_bound"]
        self.up = kw["upper_bound"]
        self.fun = kw["fun"]
        self.i_count = kw["island_count"]
        self.pop_size = kw["pop_size"]
        self.db_path = kw["db_path"]
        self.nd = 500
        self.island_local_fit = []
        self.neibor = self.get_neibor(self.i_count)
        self.all_sample = self.init_sample()
        self.pop = self.init_pop()
        self.is_data = {}
        self.is_sub = {}
        self.surros = {}
        self.rmses = {}
        self.__init()

    def get_neibor(self, i_count):
        neibor = np.zeros((i_count, 4), dtype=np.int32)
        topo_col = math.sqrt(int(i_count))
        for i in range(i_count):
            up = (i - topo_col) % i_count
            down = (i + topo_col) % i_count
            left = (i - 1) if i % topo_col != 0 else (i + topo_col - 1)
            right = (i + 1) if (i + 1) % topo_col != 0 else (i + 1 - topo_col)
            neibor[i, :] = [up, down, left, right]
        return neibor

    def __init(self):
        self.init_subsample()
        self.init_surrogate()
        self.init_local_fit()

    def get_neibor_index(self, i):
        return self.neibor[i]

    def init_sample(self):
        x = sampling.lhs(
            n=self.nd,
            d=self.d,
            lower_bound=self.low,
            upper_bound=self.up
        )
        y = [self.fun(data) for data in x]
        init_data = np.column_stack((x, y))

        return init_data

    def init_subsample(self):
        all_index = list(range(self.nd))
        for i in range(self.i_count):
            sample_idx = np.random.choice(self.nd, int(self.nd *3 / 4), replace=False)
            sub_index = list(set(all_index).difference(set(sample_idx)))
            self.is_data[i] = self.all_sample[sample_idx]
            self.is_sub[i] = self.all_sample[sub_index]
    def init_pop(self):
        all_pop = sampling.lhs(
            n=self.pop_size * self.i_count,
            d=self.d,
            lower_bound=self.low,
            upper_bound=self.up
        )
        pop_slice = np.vsplit(all_pop, self.i_count)
        return pop_slice


    def init_surrogate(self):
        for i in range(self.i_count):
            train_data = self.is_data[i]
            datax = train_data[:, :-1]
            datay = train_data[:, -1]
            surrogate = RBFN(int(np.sqrt(int(self.nd * 3 / 4))))
            surrogate.fit(datax, datay)
            surrogate.proc_id = i
            test_data = self.is_sub[i]
            tx = test_data[:, :-1]
            ty = test_data[:, -1]
            mse = mean_squared_error(ty, surrogate.predict(tx))
            rmse = np.sqrt(mse)
            self.surros[i] = surrogate
            self.rmses[i] = rmse

    def init_local_fit(self):
        for i in range(self.i_count):
            pop_i = self.pop[i]
            surrogate_i = self.surros[i]
            self.island_local_fit.append(surrogate_i.predict(pop_i))

    def get_neibor_model_rmse(self, neibor):
        surro, mse = [], []
        for nb in neibor:
            surro.append(self.surros[nb])
            mse.append(self.rmses[nb])
        return surro, mse
    def update_local_fit(self, now_fit):
        self.island_local_fit = deepcopy(now_fit)

    def update_all_fit(self):
        fit = []
        for i in range(self.i_count):
            pop_i = self.pop[i]
            surrogate_i = self.surros[i]
            fit.append(surrogate_i.predict(pop_i))
        return fit

    def get_all_surrogate(self):
        surro = []
        for i in range(self.i_count):
            surro.append(self.surros[i])
        return surro

    def get_all_rmse(self):
        rmse = []
        for i in range(self.i_count):
            rmse.append(self.rmses[i])
        return rmse

    def update_pop(self, i, pop):
        self.pop[i] = pop.copy()

    def get_pop(self, i):
        return self.pop[i]

    def get_surro(self, i):
        return self.surros[i]


    def update_surro(self, i, surro):
        self.surros[i] = surro

    def update_rmse(self, i, rmse):
        self.rmses[i] = rmse

    def get_island_sample(self, i):
        return self.is_data[i]

    def get_sub(self, i):
        return self.is_sub[i]
    def get_local_fit(self, i):
        return self.island_local_fit[i]


