import crossover
import mutation
import selection
import numpy as np

def ea(pop, config, surros, rmses):
    pop1 = crossover.sbx(
        pop=pop,
        lower_bound=config["lower_bound"],
        upper_bound=config["upper_bound"],
        pc=config["pc"],
        eta_c=config["eta_c"]
    )
    pop2 = mutation.poly_mutation(
        pop=np.vstack((pop, pop1)),
        lower_bound=config["lower_bound"],
        upper_bound=config["upper_bound"],
        pm=config["pm"],
        eta_m=config["eta_m"]
    )

    new_pop = np.vstack((pop, pop1, pop2))
    new_pop = np.unique(new_pop, axis=0)  # 去除重复行
    pop_idx, best_idx = selection.select(
        pop_size = config["pop_size"],
        pop=new_pop,
        surros = surros,
        rmses = rmses
    )
    # update the population
    pop = new_pop[pop_idx]
    best_p = new_pop[best_idx]
    return pop, best_p

def ea_w_e(pop, config, i_surro):
    pop1 = crossover.sbx(
        pop=pop,
        lower_bound=config["lower_bound"],
        upper_bound=config["upper_bound"],
        pc=config["pc"],
        eta_c=config["eta_c"]
    )
    pop2 = mutation.poly_mutation(
        pop=np.vstack((pop, pop1)),
        lower_bound=config["lower_bound"],
        upper_bound=config["upper_bound"],
        pm=config["pm"],
        eta_m=config["eta_m"]
    )

    new_pop = np.vstack((pop, pop1, pop2))
    new_pop = np.unique(new_pop, axis=0)  # 去除重复行
    pop_idx, best_idx = selection.select_w_e(
        pop_size = config["pop_size"],
        pop=new_pop,
        i_surro = i_surro,
    )
    # update the population
    pop = new_pop[pop_idx]
    best_p = new_pop[best_idx]
    return pop, best_p