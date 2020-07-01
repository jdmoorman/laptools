import argparse

import numpy as np
import pyperf
from utils import (
    geometric_matrix,
    machol_wien_matrix,
    random_machol_wien_matrix,
    uniform_matrix,
)


def get_solvers():
    from lapjv import lapjv as lapjv_lap

    # from lapjv_noinit import lapjv as lapjv_noinit_lap
    from lapsolver import solve_dense
    from scipy.optimize import linear_sum_assignment as scipy_lap
    from laptools.lap import solve as laptools_lap

    return {
        "scipy": scipy_lap,
        "lapjv": lapjv_lap,
        # "lapjv_noinit": lapjv_noinit_lap,
        "lapsolver": solve_dense,
        "laptools": laptools_lap,
    }


def time_func(n_inner_loops, solver, shape, type):
    # Note: If no matrix type is indicated, then the matrix is uniformly random
    if type == "uniform":
        cost_matrix = uniform_matrix(shape)
    if type == "geometric":
        cost_matrix = geometric_matrix(shape)
    elif type == "MW":
        cost_matrix = machol_wien_matrix(shape)
    elif type == "random_MW":
        cost_matrix = random_machol_wien_matrix(shape)
    else:
        cost_matrix = np.random.random(shape)

    t0 = pyperf.perf_counter()
    for i in range(n_inner_loops):
        solver(cost_matrix)
    return pyperf.perf_counter() - t0


def get_bench_name(size, type, solver_name):
    return "{}-{}-{}".format(size, type, solver_name)


def parse_args(benchopts):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-size-pow",
        type=int,
        metavar="POW",
        default=1,
        help="Smallest test matrix will be size 2^POW by 2^POW.",
    )
    parser.add_argument(
        "--max-size-pow",
        type=int,
        metavar="POW",
        default=2,
        help="Largest test matrix will be size 2^POW by 2^POW.",
    )
    parser.add_argument(
        "--matrix-type",
        type=str,
        metavar="X",
        default="random",
        help="The matrix is of type X.",
    )
    return parser.parse_args(benchopts)


def add_cmdline_args(cmd, args):
    cmd.append("--")
    cmd.extend(args.benchopts)


def main():
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument("benchopts", nargs="*")
    args = parse_args(runner.parse_args().benchopts)

    solvers = get_solvers()
    sizes = 2 ** np.arange(args.min_size_pow, args.max_size_pow + 1)
    type = args.matrix_type
    for size in sizes:
        for solver_name, solver_func in solvers.items():
            bench_name = get_bench_name(size, type, solver_name)
            runner.bench_time_func(
                bench_name, time_func, solver_func, (size, size), type
            )


if __name__ == "__main__":
    main()
