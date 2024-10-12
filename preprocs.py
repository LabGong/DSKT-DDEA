
def procs_params(kw):
    """update the configuration
    """
    # init configuration
    config = {
        "dim": None,                  # dimension. If not provided, it will be got from `config["samples"]` or `config["pop"]`
        "lower_bound": None,          # the lower bound of the decision variables
        "upper_bound": None,          # the upper bound of the decision variables
        "pc": 1,                      # probality of crossover
        "pm": None,                   # probality of mutation, default 1 / d
        "eta_c": 15,                  # distribution index of simulated binary crossover
        "eta_m": 15,                  # distribution index of polynomial mutation
        "island_count": 36,
        "pop_size": 100,
        "trails": 3,
        "fun": None,
        "im_interval": 10,
        "im_times": 50,
    }
    config.update(kw)
    config["pm"] = 1 / config["dim"]
    return config
