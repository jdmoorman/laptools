import numpy as np

from _augment import _solve, augment


def solve(cost_matrix, maximize=False):
    """Solve the linear sum assignment based on the cost matrix. The return
       value is the same as scipy.optimize.linear_sum_assignment.

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

    if not (
        np.issubdtype(cost_matrix.dtype, np.number)
        or cost_matrix.dtype == np.dtype(np.bool)
    ):
        raise ValueError(
            "expected a matrix containing numerical entries, got %s"
            % (cost_matrix.dtype,)
        )

    if maximize:
        cost_matrix = -cost_matrix

    if np.any(np.isneginf(cost_matrix) | np.isnan(cost_matrix)):
        raise ValueError("matrix contains invalid numeric entries")

    cost_matrix = cost_matrix.astype(np.double)
    a = np.arange(np.min(cost_matrix.shape))

    # If the cost_matrix has more rows than columns
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        # Here, col4row holds the rows in cost_matrix that are in the assignment
        _, col4row, _, _ = _solve(cost_matrix.T)

        # Sort the row indexes in the assignment
        idx_sorted = np.argsort(col4row)

        return (col4row[idx_sorted], a[idx_sorted])
    # If the cost_matrix has more columns than rows
    else:
        _, col4row, _, _ = _solve(cost_matrix)
        return (a, col4row)


def solve_lsap_with_removed_row(
    cost_matrix, row_removed, row4col, col4row, u, v, modify_val=True
):
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
    modify_val : bool, optional
        A flag that indicates whether variables are modified in place.
    """
    # cost_matrix = np.array(cost_matrix)
    n_rows, n_cols = cost_matrix.shape

    # Copy the variables if they are not modified in place.
    if not modify_val:
        row4col, col4row, u, v = row4col.copy(), col4row.copy(), u.copy(), v.copy()

    # If the freed up column does not contribute to lowering the costs of any
    # other rows, simply return the current assignments.
    freed_col = col4row[row_removed]
    freed_col_costs = cost_matrix[:, freed_col]
    if np.all(freed_col_costs >= cost_matrix[np.arange(n_rows), col4row]):
        return row4col, col4row, u, v

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

    return row4col, col4row, u, v


def solve_lsap_with_removed_col(
    cost_matrix, col_removed, row4col, col4row, u, v, modify_val=True
):
    """Solve the sub linear sum assignment problem with one column removed.

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
        return row4col, col4row, u, v

    # Copy the variables if they are not modified in place.
    if not modify_val:
        row4col, col4row, u, v = row4col.copy(), col4row.copy(), u.copy(), v.copy()

    # Update the cost matrix to reflect the column removal.
    # Create a copy of the cost matrix and update all costs associated with
    # col_removed to be infinity. We don't modify the original cost values.
    # Note: need to make sure that cost_matrix has dtype float.
    cost_matrix_copy = cost_matrix.copy()
    cost_matrix_copy[:, col_removed] = np.inf

    # Removed the assignment associated with the removed column.
    col4row[row_freed] = -1
    row4col[col_removed] = -1

    # Perform another augmenting step
    augment(cost_matrix_copy, row_freed, row4col, col4row, u, v)

    return row4col, col4row, u, v
