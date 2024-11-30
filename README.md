# Island-Based Evolutionary Computation with Diverse Surrogates and Adaptive Knowledge Transfer for High-Dimensional Data-Driven Optimization

This repository contains the implementation of the paper:  Xian-Rong Zhang, Yue-Jiao Gong, Zhiguang Cao, and Jun Zhang. 2024. Island-Based Evolutionary Computation
with Diverse Surrogates and Adaptive Knowledge Transfer for High-Dimensional Data-Driven Optimization.
ACM Transactions on Evolutionary Learning and Optimization.

## Running the Code

To run the code on the CEC2010 benchmark, use the following command:
```bash
python example_cec2010.py --fun F1 --dim 100
```

## Modifying Hyperparameters

You can modify some hyperparameters in the `preprocs.py` file. For example:
- `"im_interval": 10` means that internal optimization on the island occurs every 10 generations.
- `"im_times": 50` means that migration occurs up to 50 times.

## Adjusting Maximum Fitness Evaluations

In the `db.py` file, the `DB` class has a parameter `self.nd = 500`, which sets the maximum number of fitness evaluations to 500. You can change this to another value, such as 600.

```python
# Example modification in db.py
self.nd = 600
```

## Citation

If you use this code in your research, please cite the following paper:
```
@article{10.1145/3700886,
author = {Zhang, Xian-Rong and Gong, Yue-Jiao and Cao, Zhiguang and Zhang, Jun},
title = {Island-Based Evolutionary Computation with Diverse Surrogates and Adaptive Knowledge Transfer for High-Dimensional Data-Driven Optimization},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3700886},
doi = {10.1145/3700886},
abstract = {In recent years, there has been a growing interest in data-driven evolutionary algorithms (DDEAs) employing surrogate models to approximate the objective functions with limited data. However, current DDEAs are primarily designed for lower-dimensional problems and their performance drops significantly when applied to large-scale optimization problems (LSOPs). To address the challenge, this paper proposes an offline DDEA named DSKT-DDEA. DSKT-DDEA leverages multiple islands that utilize different data to establish diverse surrogate models, fostering diverse subpopulations and mitigating the risk of premature convergence. In the intra-island optimization phase, a semi-supervised learning method is devised to fine-tune the surrogates. It not only facilitates data argumentation, but also incorporates the distribution information gathered during the search process to align the surrogates with the evolving local landscapes. Then, in the inter-island knowledge transfer phase, the algorithm incorporates an adaptive strategy that periodically transfers individual information and evaluates the transfer effectiveness in the new environment, facilitating global optimization efficacy. Experimental results demonstrate that our algorithm is competitive with state-of-the-art DDEAs on problems with up to 1000 dimensions, while also exhibiting decent parallelism and scalability. Our DSKT-DDEA is open-source and accessible at: https://github.com/LabGong/DSKT-DDEA.},
note = {Just Accepted},
journal = {ACM Trans. Evol. Learn. Optim.},
month = nov,
keywords = {Data-driven evolutionary algorithm, large-scale optimization problems, diverse surrogate models, semi-supervised learning, adaptive knowledge transfer}
}
```


