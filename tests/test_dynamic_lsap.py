"""
   isort:skip_file
"""

import random

import numpy as np
from scipy.optimize import linear_sum_assignment

from laptools._util import one_hot
from laptools import lap


def test_solve_lsap_with_removed_row():
    """Tests for solving linear sum assignments with one row removed."""
    num_rows = 10
    num_cols = 500
    num_rounds = 100

    for i in range(num_rounds):
        # Note that here we set all costs to integer values, which might
        # lead to existence of multiple solutions.
        cost_matrix = np.random.randint(10, size=(num_rows, num_cols))
        cost_matrix = cost_matrix.astype(np.double)

        removed_row = random.randint(0, num_rows - 1)
        row_idx_1, col_idx_1 = linear_sum_assignment(cost_matrix)

        # Get the submatrix with the removed row
        sub_cost_matrix = cost_matrix[~one_hot(removed_row, num_rows), :]
        sub_row_idx_1, sub_col_idx_1 = linear_sum_assignment(sub_cost_matrix)

        # Solve the problem with dynamic algorithm
        row4col, col4row, u, v = lap._solve(cost_matrix)
        assert (
            np.array_equal(col_idx_1, col4row)
            or cost_matrix[row_idx_1, col_idx_1].sum()
            == cost_matrix[row_idx_1, col4row].sum()
        )

        lap.solve_lsap_with_removed_row(
            cost_matrix, removed_row, row4col, col4row, u, v
        )
        assert (
            np.array_equal(sub_col_idx_1, col4row[~one_hot(removed_row, num_rows)])
            or sub_cost_matrix[sub_row_idx_1, sub_col_idx_1].sum()
            == cost_matrix[
                ~one_hot(removed_row, num_rows),
                col4row[~one_hot(removed_row, num_rows)],
            ].sum()
        )


def test_solve_lsap_with_removed_col():
    """Tests for solving linear sum assignments with one column removed."""
    num_rows = 10
    num_cols = 20
    num_rounds = 1000

    for i in range(num_rounds):
        cost_matrix = np.random.randint(10, size=(num_rows, num_cols))
        cost_matrix = cost_matrix.astype(np.double)

        row_idx_1, col_idx_1 = linear_sum_assignment(cost_matrix)
        # Note that here we specifically pick a column that appears in the
        # previous optimal assignment.
        removed_col = random.choice(col_idx_1)

        # Get the submatrix with the removed col
        sub_cost_matrix = cost_matrix[:, ~one_hot(removed_col, num_cols)]
        sub_row_idx_1, sub_col_idx_1 = linear_sum_assignment(sub_cost_matrix)
        sub_cost_matrix_sum = sub_cost_matrix[sub_row_idx_1, sub_col_idx_1].sum()
        for i in range(len(sub_col_idx_1)):
            if sub_col_idx_1[i] >= removed_col:
                # Need to increment 1 to return these to their original index
                sub_col_idx_1[i] += 1

        # Solve the problem with dynamic algorithm
        row4col, col4row, u, v = lap._solve(cost_matrix)
        assert (
            np.array_equal(col_idx_1, col4row)
            or cost_matrix[row_idx_1, col_idx_1].sum()
            == cost_matrix[row_idx_1, col4row].sum()
        )

        lap.solve_lsap_with_removed_col(
            cost_matrix, removed_col, row4col, col4row, u, v
        )
        assert (
            np.array_equal(sub_col_idx_1, col4row)
            or sub_cost_matrix_sum == cost_matrix[row_idx_1, col4row].sum()
        )
