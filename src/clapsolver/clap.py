import numpy as np
from scipy.optimize import linear_sum_assignment as lap

from ._util import one_hot


def cost(i, j, costs):
    """Compute the total cost of a constrained linear assignment problem.

    Parameters
    ----------
    i : int
        Row index corresponding to the constraint.
    j : int
        Column index corresponding to the constraint.
    costs : 2darray
        A matrix of costs.

    Returns
    -------
    float
        The total cost of the linear assignment problem solution under the
        constraint that row i is assigned to column j.

    Examples
    --------
    >>> costs = [[0, 0],
    ...          [1, 0]]
    >>> clap_cost(0, 1, costs)  # Row 0 is constrained to column 1.
    1
    """
    costs = np.array(costs)
    n_rows, n_cols = costs.shape

    # Cost matrix omitting the row and column corresponding to the constraint.
    sub_costs = costs[~one_hot(i, n_rows), :][:, ~one_hot(j, n_cols)]

    # Lsap solution for the submatrix.
    sub_row_ind, sub_col_ind = lap(sub_costs)

    # Total cost is that of the submatrix lsap plus the cost of the constraint.
    return sub_costs[sub_row_ind, sub_col_ind].sum() + costs[i, j]
