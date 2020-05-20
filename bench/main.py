# from munkres import Munkres
# from scipy.optimize import linear_sum_assignment
# # from hungarian import lap # Author recommends scipy.
# from lap import lapjv
#
# from lapsolver import solve_dense
#


def run_scipy():
    from scipy.optimize import linear_sum_assignment

    return True


def run_lapjv():
    from lapjv import lapjv

    return True


def run_laptools():
    from laptools.dynamic_lsap import linear_sum_assignment

    return True


if __name__ == "__main__":
    pass
