import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import os
import sys
import main
import multiprocessing as mp
import pandas as pd
import argparse
import time
import os
from utils import get_root
import opfunu

root = os.path.realpath(os.path.dirname(__file__))
sys.path.append(root)
converg = os.path.join(root, "converg", "1000")
if not os.path.isdir(converg):
    os.makedirs(converg)

convergens = os.path.join(root, "converg", "fit")
if not os.path.isdir(convergens):
    os.makedirs(convergens)

def calculate_xmin_xmax(func_str):
    if func_str in ['F1', 'F4', 'F7', 'F8', 'F9', 'F12', 'F13', 'F14', 'F17', 'F18', 'F19', 'F20']:
        xmin = -100
        xmax = 100
    elif func_str in ['F2', 'F5', 'F10', 'F15']:
        xmin = -5
        xmax = 5
    elif func_str in ['F3', 'F6', 'F11', 'F16']:
        xmin = -32
        xmax = 32
    else:
        xmin = None
        xmax = None
    return xmin, xmax

def run_benchmark(func, dim, trails, total_run, fun_name):
    mp.set_start_method("spawn")
    lower_bound, upper_bound = calculate_xmin_xmax(fun_name)
    data = {}
    data["fit"] = []
    data["time"] = []

    for run in range(total_run):
        db_path = os.path.join(get_root(), "chkpt", func.name.split(":")[0] + "_" + str(dim), str(run))
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        time1 = time.time()
        fit, pop= main.main(
            fun=func.evaluate,
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
        print(f'\n{run}: total time = {time2 - time1}\n, minimum fitness: {fit}')
        print(f"The best pop: {pop}")
        data["fit"].append(fit)
        data["time"].append(time2 - time1)
    df = pd.DataFrame(data)
    df.to_csv(f'{converg}/{func.name.split(":")[0]}_d={dim}_trail={trails}.txt', index=False, sep="\t")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fun", default="F1", type=str)
    parser.add_argument("--dim", default=10, type=int)
    parser.add_argument("--trails", default=3, type=int)
    args = parser.parse_args()
    arg_fun = opfunu.get_functions_by_classname(args.fun + "2010")[0](ndim=args.dim)
    run_benchmark(func=arg_fun,dim=args.dim,trails=args.trails, total_run=1, fun_name=args.fun)














