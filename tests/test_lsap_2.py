import numpy as np
from numpy.testing import assert_array_equal
from scipy.optimize import linear_sum_assignment as lap

from clapsolver.dynamic_lsap import linear_sum_assignment


def test_linear_sum_assignment_against_scipy():
    for i in range(100):
        cost_matrix = np.random.rand(5, 10)

        row_ind_1, col_ind_1 = linear_sum_assignment(cost_matrix)
        row_ind_2, col_ind_2 = lap(cost_matrix)

        assert_array_equal(row_ind_1, row_ind_2)
        assert_array_equal(col_ind_1, col_ind_2)

        cost_matrix = cost_matrix.T

        row_ind_1, col_ind_1 = linear_sum_assignment(cost_matrix)
        row_ind_2, col_ind_2 = lap(cost_matrix)

        assert_array_equal(row_ind_1, row_ind_2)
        assert_array_equal(col_ind_1, col_ind_2)
