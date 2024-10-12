import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import os
import sys
import main
import benchmark
import multiprocessing as mp
import pandas as pd
import argparse
import time
import os
from utils import get_root
root = os.path.realpath(os.path.dirname(__file__))
sys.path.append(root)
converg = os.path.join(root, "converg", "1000")
if not os.path.isdir(converg):
    os.makedirs(converg)

convergens = os.path.join(root, "converg", "fit")
if not os.path.isdir(convergens):
    os.makedirs(convergens)

def run_benchmark(func, dim, trails, total_run):
    mp.set_start_method("spawn")
    lower_bound, upper_bound = benchmark.get_bound(func)
    data = {}
    data["fit"] = []
    data["time"] = []

    for run in range(total_run):
        db_path = os.path.join(get_root(), "chkpt", func.__name__ + "_" + str(dim), str(run))
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        time1 = time.time()
        fit, island_convers = main.main(
            fun=func,
            dim=dim,
            total_run=total_run,
            run=run+1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            seed=run,
            db_path = db_path,
            trails = trails
        )
        time2 = time.time()
        print(f'\n{run}: total time = {time2 - time1}\n')
        data["fit"].append(fit)
        data["time"].append(time2 - time1)
        df_converg = pd.DataFrame(island_convers)
        df_converg.to_csv(f'{convergens}/{func.__name__}_d={dim}_{run}_trail={trails}.txt', index=False, sep="\t")
    df = pd.DataFrame(data)
    df.to_csv(f'{converg}/{func.__name__}_d={dim}_trail={trails}.txt', index=False, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fun", default="ackley", type=str)
    parser.add_argument("--dim", default=100, type=int)
    parser.add_argument("--trails", default=2, type=int)
    args = parser.parse_args()
    arg_fun = None
    if args.fun == "ackley":
        arg_fun = benchmark.ackley
    elif args.fun == "rosenbrock":
        arg_fun = benchmark.rosenbrock
    elif args.fun == "ellipsoid":
        arg_fun = benchmark.ellipsoid
    elif args.fun == "griewank":
        arg_fun = benchmark.griewank
    else:
        arg_fun = benchmark.rastrigin
    run_benchmark(func=arg_fun,dim=args.dim,trails=args.trails, total_run=20)










