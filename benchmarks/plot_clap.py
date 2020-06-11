import argparse

import matplotlib.pyplot as plt
import numpy as np
import pyperf


def get_solver_from_bench(bench):
    """Extract the solver from a benchmark.
    Assumes each benchmark's name is of the format:
        "n_rows n_cols solver_name"
    """
    _, solver = bench.get_name().split("-")
    return solver


def get_size_from_bench(bench):
    """Extract the problem size from a benchmark.
    Assumes each benchmark's name is of the format:
        "n_rows n_cols solver_name"
    """
    size, _ = bench.get_name().split("-")
    n_rows, n_cols = size.split("x")
    return (int(n_rows), int(n_cols))


def get_solver_to_benches(suite):
    """Get a map from each solver to a list of its benchmarks."""
    solver_to_benches = {}
    for bench in suite:
        solver = get_solver_from_bench(bench)
        if solver not in solver_to_benches:
            solver_to_benches[solver] = []
        solver_to_benches[solver].append(bench)
    return solver_to_benches


def get_data_from_benches(benches):
    """Extract the problem size and runtimes from each benchmark."""
    sizes = [get_size_from_bench(bench) for bench in benches]
    times = [bench.get_values() for bench in benches]
    return np.array(sizes), np.array(times)


# TODO: can't do a loglog plot
def plot_suite(suite):
    """Plot the performance of each solver."""
    solver_to_benches = get_solver_to_benches(suite)
    for solver, benches in solver_to_benches.items():
        sizes, times = get_data_from_benches(benches)
        medians = np.median(times, axis=1)
        sizes_str = ["{}x{}".format(n_rows, n_cols) for n_rows, n_cols in sizes]
        plt.plot(sizes_str, medians, label=solver)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("suitefile")
    parser.add_argument("outputfile")
    return parser.parse_args()


def main():
    args = parse_args()
    suite = pyperf.BenchmarkSuite.load(args.suitefile)
    plot_suite(suite)
    plt.legend()
    plt.xlabel("Problem size")
    plt.ylabel("Time to solve")
    plt.savefig(args.outputfile)


if __name__ == "__main__":
    main()
