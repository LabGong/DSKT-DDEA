from multiprocessing import Process
import numpy as np
from evolution import ea
from utils import get_pseudo
from sklearn.metrics import mean_squared_error
from db import DB
class Island(Process):
    def __init__(self, **kw):
        super(Island, self).__init__()
        self.proc_id = kw["index"]
        self.name = "Island-" + str(self.proc_id)
        self.config = kw["config"]
        self.pop = kw["pop"]
        self.elit_generation = []
        self.neibor = kw["neibor_index"]
        self.share_surro = kw["share_surro"]
        self.share_rmse = kw["share_rmse"]
        self.island_sample = kw["samples"]
        self.sub_sample = kw["sub"]
        self.lock = kw["lock"]

        train_data = self.island_sample
        self.datax = train_data[:, :-1]
        self.datay = train_data[:, -1]
        test_data = self.sub_sample
        self.testx = test_data[:, :-1]
        self.testy = test_data[:, -1]
        self.surro = self.share_surro[self.proc_id]

        self.share_pop = kw["share_pop"]
        self.share_elit = kw["share_elit"]

    def __get_neibor_model_rmse(self):
        nb_model, nb_rmse = [],[]
        for nb in self.neibor:
            nb_model.append(self.share_surro[nb])
            nb_rmse.append(self.share_rmse[nb])
        return nb_model, nb_rmse

    def __update_rmse(self):
        pre = self.surro.predict(self.testx)
        rmse = np.sqrt(mean_squared_error(self.testy, pre))
        return rmse

    def run(self):
        # 把一些信息通过队列返回
        for i in range(self.config["im_interval"]):
            neibor_model, neibor_rmse = self.__get_neibor_model_rmse()
            new_x, new_y = get_pseudo(self.pop, neibor_model, neibor_rmse)
            new_data_x = np.row_stack((self.datax, new_x))
            new_data_y = np.append(self.datay, new_y)
            self.surro.fit(new_data_x, new_data_y)

            local_rmse = self.__update_rmse()
            with self.lock:
                self.share_surro[i] = self.surro
                self.share_rmse[i] = local_rmse

            pop = self.pop.copy()
            neibor_model.append(self.surro)
            neibor_rmse.append(local_rmse)
            next_pop, best_ind = ea(pop, self.config, neibor_model, neibor_rmse)
            self.elit_generation.append(best_ind)
            self.pop = next_pop

        self.share_elit[self.proc_id] = self.elit_generation[-1]
        self.share_pop[self.proc_id] = self.pop
