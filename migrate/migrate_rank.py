import numpy as np
import math
class Migrate(object):
    def __init__(self, island_num, neibor):
        self.island_num = island_num
        #邻居序号存入表中
        self.neibor = neibor
        self.neibor_count = neibor.shape[1]
        self.rou = 0.1
        # 从i岛到j岛的吸引力
        self.T = np.ones((self.island_num, self.neibor_count)) * (math.pow(10, -4))
        # 从i岛到j岛迁移的个体的个数
        self.migrant_ij = np.zeros((self.island_num, self.neibor_count))
        # 衰减因子
        self.rou = 0.1
        # yita
        self.yita = np.zeros((self.island_num,self.neibor_count))
        # p
        self.p = np.zeros((self.island_num, self.neibor_count))
        # 迁移率
        self.mr = 0.1
        # 从i岛到j岛迁移的个体的适应度排名
        rank = np.ones((self.island_num, self.neibor_count))
        self.rank = rank

    def get_rank(self, _des, pop, db):
        all_pop = np.vstack((pop, db.get_pop(_des)))
        fit = db.get_surro(_des).predict(np.array((all_pop)))
        pop_rank = np.argsort(fit)
        rank = np.argwhere(pop_rank == 0).reshape(-1)
        return rank

    def get_table_col_index(self, source_id, des_id):
        table_row = self.neibor[source_id]
        col_index = np.argwhere(table_row == des_id).reshape(-1)
        return col_index

    def all_scale(self, c):
        # 所有元素最大最小放缩
        return ((c - np.min(c)) / (np.max(c) - np.min(c))).copy()

    def compute_p(self, pre_fit, now_fit, db):
        fit_vary = []
        for fpre, fnow in zip(pre_fit, now_fit):
            fit_vary.append(np.mean(fpre) - np.mean(fnow))
        fit_vary_norm = self.all_scale(np.array(fit_vary))
        #因子1,每个值取值[0~1]
        fit_vary_array = np.zeros((self.island_num, self.neibor_count))
        for i in range(self.island_num):
            for j in range(self.neibor_count):
                island_id = int(self.neibor[i, j])
                fit_vary_array[i, j] = fit_vary_norm[island_id]

        #因子2
        island_surrogate_var = np.zeros((self.island_num, self.neibor_count))
        for i in range(self.island_num):
            pop = db.get_pop(i)
            local_fit = db.get_local_fit(i)
            for j in range(self.neibor_count):
                island_id = int(self.neibor[i, j])

                other_fit = db.get_surro(island_id).predict(pop)
                island_surrogate_var[i, j] = np.mean(np.abs(local_fit - other_fit)).tolist()
        island_surrogate_var = self.all_scale(island_surrogate_var)

        self.T = (1 - self.rou) * self.T + fit_vary_array * self.rank
        self.yita = island_surrogate_var.copy()
        T_yita = (self.T) * self.yita
        row_sum = np.sum(T_yita, axis=1).reshape(self.island_num, -1)
        self.p = T_yita / row_sum

    def clear_migrate_ij(self):
        self.migrant_ij = np.zeros((self.island_num, self.neibor_count))
        self.rank = np.ones((self.island_num, self.neibor_count))

    def roulette(self, island_index, pop_num):
        desti_prob = self.p[island_index].copy()
        to_id = self.neibor[island_index]
        cumsum = np.cumsum(desti_prob)
        destinations = []
        des_island = []
        for pop in range(pop_num):
            R = np.random.uniform(0, 1)
            idx = np.argwhere(R < cumsum).reshape(-1)[0]
            destinations.append(idx)
            des_island.append(int(to_id[idx]))
        for des in destinations:
            self.migrant_ij[island_index][des] += 1
        return des_island

    def mig_process(self, db):
        receive_island = []
        des_temp_pop = {}
        for index, pop in enumerate(db.pop):
            mig_pop = self.migrate_random(pop, 0.1)
            mig_pop_num = len(mig_pop)
            des_island = self.roulette(index, mig_pop_num)
            receive_island.extend(des_island)
            for _des, _pop in zip(des_island, mig_pop):
                rank = self.get_rank(_des, _pop.copy(), db)
                if _des not in des_temp_pop.keys():
                    des_temp_pop[_des] = []
                des_temp_pop[_des].append(_pop)
                self.rank[index][self.get_table_col_index(index, _des)] += (100 - rank)
        column_sum = np.sum(self.rank, axis=0) + 0.00001
        self.rank = 1 + self.rank / column_sum
        # 全部迁移结束，移民正式纳入本地种群，防止迁过来又迁走
        receive_island = list(set(receive_island))
        for ris in receive_island:
            self.update_migerate(ris, des_temp_pop[ris], db)

    def migrate_random(self, pop, mr):
        pop_size = len(pop)
        m_num = round(pop_size * mr)
        number_of_rows = pop.shape[0]
        random_indices = np.random.choice(number_of_rows,
                                          size=m_num,
                                          replace=False)
        choice_pop = pop[random_indices, :]
        migrate = np.copy(choice_pop)
        assert not choice_pop is migrate
        return migrate

    def update_migerate(self, ris, temp_pop, db):
        result = np.vstack((db.get_pop(ris), np.array(temp_pop)))
        db.update_pop(ris, result)

