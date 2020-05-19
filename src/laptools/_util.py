"""Functions for solving variants of the linear assignment problem."""
import numpy as np


def one_hot(idx, length):
    """Return a 1darray of False with a single True in the idx'th entry.

    Example
    -------
    >>> one_hot(0, 3)
    array([ True, False, False])

    Parameters
    ----------
    idx : int
        Index of the desired nonzero element.
    length : int
        Length of the resulting array.

    Returns
    -------
    1darray(bool)
        A one-hot array of False values with a True in the idx'th entry.
    """
    one_hot = np.zeros(length, dtype=np.bool)
    one_hot[idx] = True
    return one_hot
