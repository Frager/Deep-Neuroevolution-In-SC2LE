import numpy as np


class RandomUtil:
    # TODO: manage hyper parameter in a separate class
    _sigma = 0.005      # influences random number range
    _count = 1000       # number of random values to sample from (higher count is better for diversity, but slower)
    _table_seed = 123   # save these variables and use them for manager and workers

    np.random.seed(_table_seed)
    _random_table = np.random.normal(0, _sigma, _count)

    @classmethod
    def reinitialize_random_table(cls, size, sigma, seed):
        np.random.seed(seed)
        cls._random_table = np.random.normal(0, sigma, size)

    @classmethod
    def get_random_values(cls, shape, seed):
        np.random.seed(seed)
        return np.random.choice(cls._random_table, shape)
