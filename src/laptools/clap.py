import numpy as np
from scipy.optimize import linear_sum_assignment as lap

from ._util import one_hot
from .dynamic_lsap import (
    solve_lsap,
    solve_lsap_with_removed_col,
    solve_lsap_with_removed_row,
)


def cost(i, j, cost_matrix):
    """Compute the minimal cost over all constrained assignments.

    Parameters
    ----------
    i : int
        Row index corresponding to the constraint.
    j : int
        Column index corresponding to the constraint.
    cost_matrix : 2darray
        A matrix of costs.

    Returns
    -------
    float
        The total cost of the linear assignment problem solution under the
        constraint that row i is assigned to column j.
    """
    cost_matrix = np.array(cost_matrix)

    if not np.isfinite(cost_matrix[i, j]):
        return cost_matrix[i, j]

    n_rows, n_cols = cost_matrix.shape

    # Cost matrix omitting the row and column corresponding to the constraint.
    sub_cost_matrix = cost_matrix[~one_hot(i, n_rows), :][:, ~one_hot(j, n_cols)]

    # Lsap solution for the submatrix.
    sub_row_ind, sub_col_ind = lap(sub_cost_matrix)

    # Total cost is that of the submatrix lsap plus the cost of the constraint.
    return sub_cost_matrix[sub_row_ind, sub_col_ind].sum() + cost_matrix[i, j]


def costs(cost_matrix):
    """Solve a constrained linear sum assignment problem for each entry.

    The output of this function is equivalent to, but significantly more
    efficient than,

    >>> def costs(cost_matrix):
    ...     total_costs = np.empty_like(cost_matrix)
    ...     num_rows, num_cols = cost_matrix.shape
    ...     for i in range(num_rows):
    ...         for j in range(num_cols):
    ...             total_costs[i, j] = clap.cost(i, j, cost_matrix)
    ...     return total_costs

    Parameters
    ----------
    cost_matrix : 2darray
        A matrix of costs.

    Returns
    -------
    2darray
        A matrix of total constrained lsap costs. The i, j entry of the matrix
        corresponds to the total lsap cost under the constraint that row i is
        assigned to column j.
    """
    cost_matrix = np.array(cost_matrix)
    n_rows, n_cols = cost_matrix.shape
    if n_rows > n_cols:
        return costs(cost_matrix.T).T

    # Find the best lsap assignment from rows to columns without constrains.
    # Since there are at least as many columns as rows, row_idxs should
    # be identical to np.arange(n_rows). We depend on this.
    row_idxs = np.arange(n_rows)
    row4col, col4row, u, v = solve_lsap(cost_matrix)

    # Column vector of costs of each assignment in the lsap solution.
    lsap_costs = cost_matrix[row_idxs, col4row]
    lsap_total_cost = lsap_costs.sum()

    # Each row is column indexes ordered by their costs in that row.
    sorted_costs_ind = np.argsort(cost_matrix, axis=1)

    # When we add the constraint assigning row i to column j, lsap_col_idxs[i]
    # is freed up. If lsap_col_idxs[i] cannot improve on the cost of one of the
    # other row assignments, it does not need to be reassigned to another row.
    # If additionally column j is not in lsap_col_idxs, it is not taken away
    # from any of the other row assignments. In this situation, the resulting
    # total assignment costs are:
    total_costs = lsap_total_cost - lsap_costs[:, None] + cost_matrix

    for i, freed_j in enumerate(col4row):
        # When row i is constrained to another column, can column j be
        # reassigned to improve the assignment cost of one of the other rows?
        # To deal with that, we solve the lsap with row i omitted. For the
        # majority of constraints on row i's assignment, this will not conflict
        # with the constraint. When it does conflict, we fix the issue later.
        sub_ind = ~one_hot(i, n_rows)
        sub_cost_matrix = cost_matrix[sub_ind, :]

        new_row4col, new_col4row, new_u, new_v = solve_lsap_with_removed_row(
            cost_matrix, i, row4col, col4row, u, v, modify_val=False
        )

        sub_total_cost = cost_matrix[sub_ind, new_col4row[sub_ind]].sum()
        new_col4row[i] = -1  # Row i is having a constraint applied.

        # This calculation will end up being wrong for the columns in
        # lsap_col_idxs[sub_col_ind]. This is because the constraint in
        # row i in these columns will conflict with the sub assignment.
        # These miscalculations are corrected later.
        total_costs[i, :] = cost_matrix[i, :] + sub_total_cost

        # new_col4row now contains the optimal assignment columns ignoring row i.
        new_col4row[i] = np.setdiff1d(col4row, new_col4row)[0]
        total_costs[i, new_col4row[i]] = cost_matrix[row_idxs, new_col4row].sum()

        # When we solve the lsap with row i removed, we update row4col accordingly.
        sub_row4col = new_row4col.copy()
        sub_row4col[sub_row4col == i] = -1
        sub_row4col[sub_row4col > i] -= 1

        for other_i, stolen_j in enumerate(new_col4row):
            if other_i == i:
                continue

            if not np.isfinite(cost_matrix[i, stolen_j]):
                total_costs[i, stolen_j] = cost_matrix[i, stolen_j]
                continue

            # Row i steals column stolen_j from other_i because of constraint.
            new_col4row[i] = stolen_j
            new_col4row[other_i] = -1

            # Row other_i must find a new column. What is its next best option?
            best_j = sorted_costs_ind[other_i, 0]
            second_best_j = sorted_costs_ind[other_i, 1]
            next_best_j = best_j if (best_j != stolen_j) else second_best_j

            # Is the next best option available? If so, use it. Otherwise,
            # solve the constrained lsap from scratch.
            if next_best_j not in new_col4row:
                new_col4row[other_i] = next_best_j
                total_costs[i, stolen_j] = cost_matrix[row_idxs, new_col4row].sum()
            else:
                # Otherwise, solve the lsap with stolen_j removed
                # TODO: we might want to bring back potential_cols here.
                _, new_new_col4row, _, _ = solve_lsap_with_removed_col(
                    sub_cost_matrix,
                    stolen_j,
                    sub_row4col,
                    new_col4row[sub_ind],
                    new_u[sub_ind],  # dual variable associated with rows
                    new_v,  # dual variable associated with cols
                    modify_val=False,
                )

                total_costs[i, stolen_j] = (
                    cost_matrix[i, stolen_j]
                    + cost_matrix[sub_ind, new_new_col4row].sum()
                )

            # Give other_i its column back in preparation for the next round.
            new_col4row[other_i] = stolen_j
            new_col4row[i] = -1

    # For those constraints which are compatible with the unconstrained lsap:
    total_costs[row_idxs, col4row] = lsap_total_cost

    return total_costs
