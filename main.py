# -*- coding: utf-8 -*-
import dill
import tqdm
import preprocs
import multiprocessing as mp
from island import Island
from migrate.migrate_rank import Migrate
from copy import deepcopy
import numpy as np
from optimal import Optimal
from db import DB

def main(**kw):
    """
    Return:
        The best solutions in each iteration
    """
    print(f"cpu 核心数{mp.cpu_count()}.", flush=True)
    TIMEOUT = 48 * 3600
    np.random.seed(int(kw["seed"]))
    config = preprocs.procs_params(kw)
    db = DB(config)

    migrate = Migrate(config["island_count"], db.neibor)
    opt = Optimal(config["im_interval"],  config["trails"], config["island_count"])

    pre_fit = deepcopy(db.island_local_fit)
    pbar = tqdm.tqdm(range(config["im_times"]))
    dis = f'{config["fun"].__name__} {config["dim"]}d - {config["run"]}/{config["total_run"]}'
    pbar.set_description(dis)


    for im_time in pbar:
        proc = list()
        lock = mp.Manager().Lock()

        share_surro = mp.Manager().dict()
        for i, _surro in db.surros.items():
            share_surro[i] = _surro

        share_rmse = mp.Manager().dict()
        for i, _rmse in db.rmses.items():
            share_rmse[i] = _rmse

        share_pop = mp.Manager().dict()
        share_elit = mp.Manager().dict()


        for i in range(config["island_count"]):
            p = Island(
                index=i,
                config=config,
                pop=db.pop[i],
                neibor_index = db.get_neibor_index(i),
                lock = lock,
                share_surro = share_surro,
                share_rmse = share_rmse,
                samples = db.is_data[i],
                sub = db.is_sub[i],
                share_pop = share_pop,
                share_elit = share_elit,
            )
            proc.append(p)
        for p in proc:
            p.start()
        for p in proc:
            p.join(timeout=TIMEOUT)
        for p in proc:
            if p.is_alive():
                p.terminate()
                p.join()
        for p in proc:
            p.close()

        assert len(share_pop) == db.i_count
        assert len(share_elit) == db.i_count

        for i, pop in share_pop.items():
            db.update_pop(i, pop)
        for i, surro in share_surro.items():
            db.update_surro(i, surro)
        for i,rmse in share_rmse.items():
            db.update_rmse(i, rmse)
        for i, elit in share_elit.items():
            opt.update_elit(i, elit)

        now_fit = db.update_all_fit()
        db.update_local_fit(now_fit)

        all_surro = db.get_all_surrogate()
        all_rmse = db.get_all_rmse()
        opt.partial_optimal( all_surro, all_rmse)

        if not opt.is_continuable():
            print(f"{im_time} early stop !")
            break
        else:
            migrate.compute_p(pre_fit, now_fit, db)
            migrate.clear_migrate_ij()
            migrate.mig_process(db)

    all_surro = db.get_all_surrogate()
    all_rmse = db.get_all_rmse()
    pop = opt.optimal(all_surro, all_rmse)

    fit1 = config["fun"](pop).item()
    return fit1, pop
