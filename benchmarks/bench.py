import argparse

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
    for size in sizes:
        for solver_name, solver_func in solvers.items():
            bench_name = get_bench_name(size, solver_name)
            runner.bench_time_func(bench_name, time_func, solver_func, (size, size))


if __name__ == "__main__":
    main()
