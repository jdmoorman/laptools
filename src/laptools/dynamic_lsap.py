import numpy as np


def augment(cost_matrix, cur_row, row4col, col4row, u, v):
    """Find the shortest augmenting path for the selected row. Update the dual
       variables. Augment the previous solution.

    Parameters
    ----------
    cost_matrix : 2darray
         A numpy matrix of costs. Note that the number of rows cannot exceed
         the number of columns. It also cannot hold np.nan values.
    cur_row : int
        Index of the row from which the augmenting starts.
    row4col: 1darray
             Specify the rows to which each column is matched with.
    col4row: 1darray
             Specify the columns to which each row is matched with.
    u : 1darray
        The dual cost vector for rows.
    v : 1darray
        The dual cost vector for columns.
    """
    n_rows, n_cols = cost_matrix.shape

    min_val = 0
    row_idx = cur_row
    remaining = set(range(n_cols))

    path = np.full(n_cols, -1)
    shortest_path_costs = np.full(n_cols, np.inf)

    # SR: row vertices visited in the shortest augmenting path
    # SC: col vertices visited in the shortest augmenting path
    SR, SC = set(), set()

    sink = -1
    while sink == -1:
        idx_min = -1  # idx_min = argmin(shortest_path_costs)
        lowest = np.inf
        SR.add(row_idx)

        # Iterates thru all the column vertices that have not been visited
        for col_idx in remaining:
            r = min_val + cost_matrix[row_idx, col_idx] - u[row_idx] - v[col_idx]

            if r < shortest_path_costs[col_idx]:
                path[col_idx] = row_idx
                shortest_path_costs[col_idx] = r

            if shortest_path_costs[col_idx] < lowest or (
                shortest_path_costs[col_idx] == lowest and row4col[col_idx] == -1
            ):
                lowest = shortest_path_costs[col_idx]
                idx_min = col_idx

        min_val = lowest

        # If the cost matrix is infeasible
        if min_val == np.inf:
            raise ValueError("The cost matrix is infeasible.")

        if row4col[idx_min] == -1:
            sink = idx_min
        else:
            row_idx = row4col[idx_min]

        SC.add(idx_min)

        # Remove the visited column from remaining
        remaining.remove(idx_min)

    # If no augmenting path has been found
    if sink < 0:
        raise ValueError("No augmenting path has been found.")

    # Update the dual variables
    for i in SR:
        if i == cur_row:
            u[i] += min_val
        else:
            u[i] += min_val - shortest_path_costs[col4row[i]]

    for j in SC:
        v[j] -= min_val - shortest_path_costs[j]

    # Augment the Previous Solution
    col_idx = sink
    while True:
        row_idx = path[col_idx]
        row4col[col_idx] = row_idx
        col4row[row_idx], col_idx = col_idx, col4row[row_idx]
        if row_idx == cur_row:
            break


def solve_lsap(cost_matrix):
    """Solve the rectangular linear sum assignment problem via an augmenting
       path approach. Note that the cost matrix can be rectangular, but we
       require that its number of rows cannot exceed its number of columns.

    Parameters
    ----------
    cost_matrix : 2darray
         A matrix of costs. The number of rows cannot exceed the number of columns.

    Returns
    -------
    row4col: 1darray
             Specify the rows to which each column is matched with.
    col4row: 1darray
             Specify the columns to which each row is matched with.
    u : 1darray
        The dual cost vector for rows.
    v : 1darray
        The dual cost vector for columns.
    """
    cost_matrix = np.array(cost_matrix)
    n_rows, n_cols = cost_matrix.shape

    # cast the dtype of cost_matrix as double
    cost_matrix = cost_matrix.astype(np.double)

    u = np.zeros(n_rows)
    v = np.zeros(n_cols)
    col4row = np.full(n_rows, -1)
    row4col = np.full(n_cols, -1)

    # Iteratively build the solution
    for cur_row in range(n_rows):
        augment(cost_matrix, cur_row, row4col, col4row, u, v)

    return row4col, col4row, u, v


def solve_lsap_with_removed_row(cost_matrix, row_removed, row4col, col4row, u, v):
    """Solve the sub linear sum assignment problem with one row removed.

    Note: the removed row will still be assigned to a column.
    Note: While the cost matrix will not be modified, the dual variables would
          be updated as if the costs of removed row are uniformly zero.

    Parameters
    ----------
    cost_matrix : 2darray
         A matrix of costs.
    row_removed : int
         The index of the row that is to be removed.
    row4col: 1darray
             Specify the rows to which each column is matched with.
    col4row: 1darray
             Specify the columns to which each row is matched with.
    u : 1darray
        The dual cost vector for rows.
    v : 1darray
        The dual cost vector for columns.
    """
    cost_matrix = np.array(cost_matrix)
    n_rows, n_cols = cost_matrix.shape

    # Update the cost matrix to reflect the row removal.
    # Create a copy of the cost matrix and update all costs associated with
    # row_removed to be zero. We don't modify the original cost values.
    cost_matrix_copy = cost_matrix.copy()
    cost_matrix_copy[row_removed] = 0

    # Update the dual variables
    u[row_removed] = np.min(cost_matrix_copy[row_removed] - v)

    # Perform another augmenting step, only on the sub-cost-matrix that
    # involves the rows and columns in the original optimal assignment.
    # The row to be removed have all of its associated costs set to 0.
    sub_cost_matrix = cost_matrix_copy[:, col4row]
    sub_v = v[col4row]

    # Remove the assignment associated with the removed row. Note that in the
    # submatrix, the removed row and the freed up column have the same index.
    # TODO: check whether the following expressions are correct
    sub_col4row = np.arange(n_rows)
    sub_col4row[row_removed] = -1
    sub_row4col = np.arange(n_rows)
    sub_row4col[row_removed] = -1

    # Find the shortest augmenting path for the sub square lsap and augment.
    augment(sub_cost_matrix, row_removed, sub_row4col, sub_col4row, u, sub_v)

    # Update the original assignment
    # Note: update every variable that depends on col4row for indexing first.
    row4col[col4row] = sub_row4col
    v[col4row] = sub_v
    col4row[:] = col4row[sub_col4row]


def solve_lsap_with_removed_col(cost_matrix, col_removed, row4col, col4row, u, v):
    """Solve the sub linear sum assignment problem with one column removed.

    TODO: Add tests for this function.
    Note: While the cost matrix will not be modified, the dual variables would
          be updated as if the removed column does not exist.

    Parameters
    ----------
    cost_matrix : 2darray
         A matrix of costs.
    col_removed : int
         The index of the column that is to be removed.
    row4col: 1darray
             Specify the rows to which each column is matched with.
    col4row: 1darray
             Specify the columns to which each row is matched with.
    u : 1darray
        The dual cost vector for rows.
    v : 1darray
        The dual cost vector for columns.
    """
    row_freed = row4col[col_removed]

    # If the removed column is resigned to nothing
    # No need to reassign anything
    if row_freed == -1:
        return

    cost_matrix = np.array(cost_matrix)
    cost_matrix = cost_matrix.astype(np.double)

    # Update the cost matrix to reflect the column removal.
    # Create a copy of the cost matrix and update all costs associated with
    # col_removed to be infinity. We don't modify the original cost values.
    # Note: need to make sure that cost_matrix has dtype float.
    cost_matrix_copy = cost_matrix.copy()
    cost_matrix_copy[:, col_removed] = np.inf

    # TODO: do we need to update the dual variables?

    # Removed the assignment associated with the removed column.
    col4row[row_freed] = -1
    row4col[col_removed] = -1

    # Perform another augmenting step
    augment(cost_matrix_copy, row_freed, row4col, col4row, u, v)


def linear_sum_assignment(cost_matrix, maximize=False):
    """Solve the linear sum assignment based on the cost matrix. The return
       value is the same as in scipy.

    Parameters
    ----------
    cost_matrix : 2darray
         A matrix of costs.

    Returns
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.
    """
    cost_matrix = np.array(cost_matrix)

    # The following are taken from scipy implementation.
    if len(cost_matrix.shape) != 2:
        raise ValueError(
            "expected a matrix (2-d array), got a %r array" % (cost_matrix.shape,)
        )

    # fmt: off
    # TODO: the following is not passing flake8
    if not (np.issubdtype(cost_matrix.dtype, np.number) or
            cost_matrix.dtype == np.dtype(np.bool)):
        raise ValueError(
            "expected a matrix containing numerical entries, got %s"
            % (cost_matrix.dtype,)
        )
    # fmt: on

    if maximize:
        cost_matrix = -cost_matrix

    if np.any(np.isneginf(cost_matrix) | np.isnan(cost_matrix)):
        raise ValueError("matrix contains invalid numeric entries")

    # cost_matrix = cost_matrix.astype(np.double)
    a = np.arange(np.min(cost_matrix.shape))

    # If the cost_matrix has more rows than columns
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        # Here, col4row holds the rows in cost_matrix that are in the assignment
        _, col4row, _, _ = solve_lsap(cost_matrix.T)

        # Sort the row indexes in the assignment
        idx_sorted = np.argsort(col4row)

        return (col4row[idx_sorted], a[idx_sorted])
    # If the cost_matrix has more columns than rows
    else:
        _, col4row, _, _ = solve_lsap(cost_matrix)
        return (a, col4row)
