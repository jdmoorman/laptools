import numpy as np


def augment(cost_matrix, u, v, row4col, col4row, curRow):
    """Find the shortest augmenting path for the selected row. Update the dual
       variables. Augment the previous solution.

    Parameters
    ----------
    cost_matrix : 2darray
         A matrix of costs.
    u : 1darray
        The dual cost vector for rows.
    v : 1darray
        The dual cost vector for columns.
    row4col: 1darray
             Specify the rows to which each column is matched with.
    col4row: 1darray
             Specify the columns to which each row is matched with.
    curRow : int
        Index of the row from which the augmenting starts.
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
        augment(cost_matrix, u, v, row4col, col4row, curRow)

    import IPython

    IPython.embed()

    return col4row


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

    solve_lsap(cost_matrix)
