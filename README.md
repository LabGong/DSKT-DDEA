# DSKT-DDEA
# Island-Based Evolutionary Computation with Diverse Surrogates and Adaptive Knowledge Transfer for High-Dimensional Data-Driven Optimization

This repository contains the implementation of the paper:  Xian-Rong Zhang, Yue-Jiao Gong, Zhiguang Cao, and Jun Zhang. 2024. Island-Based Evolutionary Computation
with Diverse Surrogates and Adaptive Knowledge Transfer for High-Dimensional Data-Driven Optimization.
ACM Transactions on Evolutionary Learning and Optimization. 37, 4, Article 111 (August 2024), 31 page.

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
@article{your_paper,
  title={Island-Based Evolutionary Computation with Diverse Surrogates and Adaptive Knowledge Transfer for High-Dimensional Data-Driven Optimization},
  author={Xian-Rong Zhang, Yue-Jiao Gong, Zhiguang Cao, and Jun Zhang. },
  journal={ACM Transactions on Evolutionary Learning and Optimization},
  year={2024},
  volume={XX},
  number={YY},
  pages={ZZZ-ZZZ}
}
```



