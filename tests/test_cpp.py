import numpy as np

from _augment import augment


def test_augment():
    """Augment should modify its arguments inplace."""
    cost_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 5.0], [2.0, 3.0, 2.0]])
    row4col = np.array([-1, 1, 2])
    col4row = np.array([-1, 1, 2])
    u = np.array([0.0, 2.0, 2.0])
    v = np.array([-2.0, 0.0, 0.0])

    row4col_copy, col4row_copy = row4col.copy(), col4row.copy()
    inputs = [row4col, col4row, u, v]
    outputs = augment(cost_matrix, 0, row4col, col4row, u, v)

    assert [elm.dtype for elm in inputs] == [elm.dtype for elm in outputs]
    assert [id(elm) for elm in inputs] == [id(elm) for elm in outputs]

    assert row4col_copy.tolist() != row4col.tolist()
    assert col4row_copy.tolist() != col4row.tolist()
