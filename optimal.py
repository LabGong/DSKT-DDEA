import copy
import numpy as np
import math
from db import DB
from utils import get_elit
class Optimal(object):
    def __init__(self, im_interval, num_trials, icount):
        self.im_interval = im_interval
        self.is_count = icount

        self.global_opt = []
        self.global_fit = []

        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_loss = math.inf

        self.elits = {}
        for i in range(self.is_count):
            self.elits[i] = []

        self.island_convers = {}
        for i in range(self.is_count):
            self.island_convers[i] = []

    def update_elit(self, i, pop):
        self.elits[i].append(pop)

    def update_convers(self, i, fit_list):
        self.island_convers[i].extend(fit_list)

    def partial_optimal(self, all_surro, all_rmse):
        elit = np.array([e[-1] for i,e in self.elits.items()])

        opt_pop, opt_fit = get_elit(elit, all_surro, all_rmse)
        self.global_opt.append(copy.deepcopy(opt_pop))
        self.global_fit.append(opt_fit)


    def optimal(self, surro, mse):

        opt_pop, opt_fit = get_elit(np.array(self.global_opt), surro, mse)
        return copy.deepcopy(opt_pop)

    def last(self):
        return copy.deepcopy(self.global_opt[-1])
    def is_continuable(self):
        if self.global_fit[-1] < self.best_loss:
            self.best_loss = self.global_fit[-1]
            self.trial_counter = 0
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


