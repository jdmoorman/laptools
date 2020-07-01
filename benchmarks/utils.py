import random

import numpy as np


def uniform_matrix(shape, low=0.0, high=1.0):
    """Generate a uniformly random matrix of the given shape."""
    return np.random.uniform(low=low, high=high, size=shape)


def geometric_matrix(shape, low=0.0, high=1.0):
    """Generate a geometric matrix of the given shape."""
    n_rows, n_cols = shape
    A = np.random.uniform(low=low, high=high, size=(n_rows + n_cols,))
    B = np.random.uniform(low=low, high=high, size=(n_rows + n_cols,))

    A_mat = np.array(
        [[A[i] - A[n_rows + j] for j in range(n_cols)] for i in range(n_rows)]
    )
    B_mat = np.array(
        [[B[i] - B[n_rows + j] for j in range(n_cols)] for i in range(n_rows)]
    )

    return np.sqrt(A_mat ** 2 + B_mat ** 2) + 1.0


def machol_wien_matrix(shape):
    """Generate a Machol-Wien matrix of the given shape."""
    n_rows, n_cols = shape
    return np.array(
        [[i * j + 1 for j in range(1, n_cols + 1)] for i in range(1, n_rows + 1)]
    )


def random_machol_wien_matrix(shape, low=0.0, high=1.0):
    """Generate a random Machol-Wien matrix of the given shape."""
    n_rows, n_cols = shape
    return np.array(
        [
            [random.randint(1, i * j + 1) for j in range(1, n_cols + 1)]
            for i in range(1, n_rows + 1)
        ]
    )
