import numpy as np
import pyperf


def get_solvers():
    from lapjv import lapjv as lapjv_lap
    from scipy.optimize import linear_sum_assignment as scipy_lap
    from laptools.dynamic_lsap import linear_sum_assignment as laptools_lap

    return {
        "scipy": scipy_lap,
        "lapjv": lapjv_lap,
        "laptools": laptools_lap,
    }


def time_func(n_inner_loops, solver, shape):
    cost_matrix = np.random.random(shape)
    t0 = pyperf.perf_counter()
    for i in range(n_inner_loops):
        solver(cost_matrix)
    return pyperf.perf_counter() - t0


def get_bench_name(size, solver_name):
    return "{}-{}".format(size, solver_name)


def main():
    runner = pyperf.Runner()
    solvers = get_solvers()
    sizes = 2 ** np.arange(3, 11)
    for size in sizes:
        for solver_name, solver_func in solvers.items():
            bench_name = get_bench_name(size, solver_name)
            runner.bench_time_func(bench_name, time_func, solver_func, (size, size))


if __name__ == "__main__":
    main()
