# -*- coding: utf-8 -*-
import numpy as np
from utils import get_wsuro
def select(pop_size, pop, surros, rmses):
    pre_y = get_wsuro(pop, surros, rmses)
    num = np.argsort(pre_y)
    idx = num[:pop_size]
    best_idx = num[0]
    return idx,best_idx

def select_w_e(pop_size, pop, i_surro):
    pre_y = i_surro.predict(pop)
    num = np.argsort(pre_y)
    idx = num[:pop_size]
    best_idx = num[0]
    return idx,best_idx

