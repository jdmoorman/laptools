import numpy as np


def augment(cost_matrix, curRow, row4col, col4row, u, v):
    """Find the shortest augmenting path for the selected row. Update the dual
       variables. Augment the previous solution.

    Parameters
    ----------
    cost_matrix : 2darray
         A matrix of costs.
    curRow : int
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
    cost_matrix = np.array(cost_matrix)
    n_rows, n_cols = cost_matrix.shape

    minVal = 0
    row_idx = curRow
    remaining = set(range(n_cols))

    path = np.full(n_cols, -1)
    shortestPathCosts = np.full(n_cols, np.inf)

    # TODO: figure out a better way to store these vertices
    SR, SC = set(), set()

    sink = -1
    while sink == -1:
        idx_min = -1  # idx_min = argmin(shortestPathCosts)
        lowest = np.inf
        SR.add(row_idx)

        # Iterates thru all the column vertices that have not been visited
        for col_idx in remaining:
            r = minVal + cost_matrix[row_idx, col_idx] - u[row_idx] - v[col_idx]

            if r < shortestPathCosts[col_idx]:
                path[col_idx] = row_idx
                shortestPathCosts[col_idx] = r

            if shortestPathCosts[col_idx] < lowest or (
                shortestPathCosts[col_idx] == lowest and row4col[col_idx] == -1
            ):
                lowest = shortestPathCosts[col_idx]
                idx_min = col_idx

        minVal = lowest

        # If the cost matrix is infeasible
        if minVal == np.inf:
            raise Exception("The cost matrix is infeasible.")

        if row4col[idx_min] == -1:
            sink = idx_min
        else:
            row_idx = row4col[idx_min]

        SC.add(idx_min)

        # Remove the visited column from remaining
        remaining.remove(idx_min)

    # If no augmenting path has been found
    if sink < 0:
        raise Exception("No augmenting path has been found.")

    # Update the dual variables
    for i in SR:
        if i == curRow:
            u[i] += minVal
        else:
            u[i] += minVal - shortestPathCosts[col4row[i]]

    for j in SC:
        v[j] -= minVal - shortestPathCosts[j]

    # Augment the Previous Solution
    col_idx = sink
    while True:
        row_idx = path[col_idx]
        row4col[col_idx] = row_idx
        col4row[row_idx], col_idx = col_idx, col4row[row_idx]
        if row_idx == curRow:
            break

    # TODO: Figure out if we need to return anything
    print(v)


def solve_lsap(cost_matrix):
    """Solve the rectangular linear sum assignment problem via an augmenting
       path approach.

    Parameters
    ----------
    cost_matrix : 2darray
         A matrix of costs.

    Returns
    -------
    TODO: figure out what is returned
    """
    cost_matrix = np.array(cost_matrix)
    n_rows, n_cols = cost_matrix.shape

    u = np.zeros(n_rows)
    v = np.zeros(n_cols)
    col4row = np.full(n_rows, -1)
    row4col = np.full(n_cols, -1)

    # Iteratively build the solution
    for curRow in range(n_rows):
        augment(cost_matrix, curRow, row4col, col4row, u, v)

    # TODO: Figure out if we should return things in scipy format
    return row4col, col4row, u, v


def solve_sub_lsap(cost_matrix, row_removed, row4col, col4row, u, v):
    """Solve the sub linear sum assignment problem with one row removed.

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
    """
    cost_matrix = np.array(cost_matrix)
    n_rows, n_cols = cost_matrix.shape

    # Update the cost matrix to reflect the removal
    # TODO: is zero a good number?
    cost_matrix[row_removed] = 0

    # Update the dual variables
    u[row_removed] = np.min(cost_matrix[row_removed] - v)

    # Perform another augmenting step, only on the sub-cost-matrix that
    # involves the rows and columns in the original optimal assignment.
    # The row to be removed have all of its associated costs set to 0.
    sub_cost_matrix = cost_matrix[:, col4row]
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
    col4row = col4row[sub_col4row]

    return row4col, col4row, u, v


if __name__ == "__main__":
    # fmt: off
    cost_matrix = [[np.inf, np.inf, np.inf, np.inf,      3],
                   [     7, np.inf,     23, np.inf, np.inf],
                   [    17,     24, np.inf, np.inf, np.inf],
                   [np.inf,      6,     13,     20, np.inf]]
    # fmt: on

    # curRow = 1
    # u = np.array([0, 0, 17, 6])
    # v = np.array([0, 0, 0, 0, 0])
    # row4col = np.array([2, 3, -1, -1, -1])
    # col4row = np.array([-1, -1, 0, 1])
    # augment(cost_matrix, u, v, row4col, col4row, curRow)

    row4col, col4row, u, v = solve_lsap(cost_matrix)
    row4col, col4row, u, v = solve_sub_lsap(cost_matrix, 2, row4col, col4row, u, v)
