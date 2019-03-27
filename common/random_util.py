import numpy as np
import os


class RandomUtil:
    # Contains all code responsible for generating random numbers.
    # For deterministic output use np.random.seed(seed) before use
    # Used for initializing models

    _sigma = 0.005      # influences random number range
    _count = 10000       # number of random values to sample from (higher count is better for diversity, but slower)
    _table_seed = 1234   # save these variables and use them for manager and workers so everyone has the same table

    np.random.seed(_table_seed)
    _random_table = np.random.normal(0, _sigma, _count)

    @classmethod
    def reinitialize_random_table(cls, size, sigma, seed):
        np.random.seed(seed)
        cls._random_table = np.random.normal(0, sigma, size)

    @classmethod
    def set_seed(cls, seed):
        np.random.seed(seed)

    @classmethod
    def get_random_values(cls, shape):
        # Pics random values from the random number table
        # For deterministic output use np.random.seed(seed) before use
        return np.random.choice(cls._random_table, shape)

    @classmethod
    def normalized_columns_initializer(cls, shape):
        # Sets random values and normalizes them
        # For deterministic output use np.random.seed(seed) before use
        out = np.random.randn(np.prod(shape[:-1]), shape[-1]).astype(np.float32)
        out *= 1.0 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))     # normalization
        out = np.reshape(out, shape)
        return out

    @classmethod
    def get_random_seed(cls):
        # Get a random seed.
        # Used by GA to initialize models and add seeds for mutation
        random_bytes = os.urandom(4)
        seed = int.from_bytes(random_bytes, byteorder="big")
        return seed
