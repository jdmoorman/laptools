import random

import numpy as np
from numpy.testing import assert_array_equal
from scipy.optimize import linear_sum_assignment as lap

from clapsolver._util import one_hot
from clapsolver.dynamic_lsap import solve_lsap, solve_lsap_with_removed_row


def test_solve_lsap_with_removed_row():
    num_rows = 10
    num_cols = 500
    num_rounds = 100

    for i in range(num_rounds):
        cost_matrix = np.random.randint(10, size=(num_rows, num_cols))
        removed_row = random.randint(0, num_rows - 1)
        row_idx_1, col_idx_1 = lap(cost_matrix)

        # Get the submatrix with the removed row
        sub_cost_matrix = cost_matrix[~one_hot(removed_row, num_rows), :]
        sub_row_idx_1, sub_col_idx_1 = lap(sub_cost_matrix)

        # Solve the problem with dynamic algorithm
        row4col, col4row, u, v = solve_lsap(cost_matrix)
        try:
            assert_array_equal(col_idx_1, col4row)
        except AssertionError:
            print("cost_matrix:", cost_matrix)

        solve_lsap_with_removed_row(cost_matrix, removed_row, row4col, col4row, u, v)
        try:
            assert_array_equal(sub_col_idx_1, col4row[~one_hot(removed_row, num_rows)])
        except AssertionError:
            print("cost_matrix:", cost_matrix)
            print("removed_row:", removed_row)
