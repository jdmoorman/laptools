import pytest

from clapsolver import clap

# fmt: off
costs1 = [[4, 1, 3],
          [2, 0, 5],
          [3, 2, 2]]

costs2 = [[4, 1, 3, 6],
          [2, 0, 5, 7],
          [3, 2, 2, 8]]

costs3 = [[4, 1, 3],
          [2, 0, 5],
          [3, 2, 2],
          [6, 7, 8]]
# fmt: on


@pytest.mark.parametrize(
    "i, j, costs, expected",
    [
        (0, 0, costs1, 6),
        (0, 1, costs1, 5),
        (0, 2, costs1, 6),
        (1, 0, costs1, 5),
        (1, 1, costs1, 6),
        (1, 2, costs1, 9),
        (2, 0, costs1, 6),
        (2, 1, costs1, 7),
        (2, 2, costs1, 5),
        (0, 3, costs2, 8),
        (1, 3, costs2, 10),
        (2, 3, costs2, 11),
        (3, 0, costs3, 8),
        (3, 1, costs3, 11),
        (3, 2, costs3, 11),
    ],
)
def test_clap_cost(i, j, costs, expected):
    """Verify clap.cost works on small examples computable by hand."""
    assert clap.cost(i, j, costs) == expected
