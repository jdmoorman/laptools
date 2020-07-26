import numpy as np

from . import lap
from ._util import one_hot


def costs(cost_matrix):
    """An naive algorithm of solving all constraint LAPs. """
    n_rows, n_cols = cost_matrix.shape
    total_costs = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            sub_row_ind = ~one_hot(i, n_rows)
            sub_col_ind = ~one_hot(j, n_cols)
            sub_cost_matrix = cost_matrix[sub_row_ind, :][:, sub_col_ind]
            row_idx, col_idx = lap.solve(sub_cost_matrix)
            sub_total_cost = sub_cost_matrix[row_idx, col_idx].sum()
            total_costs[i, j] = cost_matrix[i, j] + sub_total_cost
    return total_costs
