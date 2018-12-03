import numpy as np
import os


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

    @classmethod
    def xavier_initializer(cls, shape, num_in, num_out, seed):
        np.random.seed(seed)
        return np.random.rand(*shape) * np.sqrt(1/(num_in+num_out))

    @classmethod
    def get_random_seed(cls):
        random_bytes = os.urandom(4)
        seed = int.from_bytes(random_bytes, byteorder="big")
        return seed
