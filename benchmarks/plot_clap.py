import argparse
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pyperf
import seaborn as sns


def get_solver_from_bench(bench):
    """Extract the solver from a benchmark.
    Assumes each benchmark's name is of the format:
        "n_rows n_cols solver_name"
    """
    _, _, solver = bench.get_name().split("-")
    return solver


def get_type_from_bench(bench):
    """Extract the matrix type from a benchmark.

    Assumes each benchmark's name is of the format:

        "problem_size matrix_type solver_name"
    """
    _, type, _ = bench.get_name().split("-")
    return type


def get_size_from_bench(bench):
    """Extract the problem size from a benchmark.
    Assumes each benchmark's name is of the format:
        "n_rows n_cols solver_name"
    """
    size, _, _ = bench.get_name().split("-")
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


# TODO: also get the 75% confidence interval
def plot_suite(suite):
    """Plot the performance of each solver."""
    solver_to_benches = get_solver_to_benches(suite)
    colors = cycle(sns.color_palette())
    line_styles = cycle(["-", "--", "-.", ":"])
    for solver, benches in solver_to_benches.items():
        sizes, times = get_data_from_benches(benches)
        # Plot the median and the 90% confidence interval
        medians = np.median(times, axis=1)
        upper = np.percentile(times, 95, axis=1)
        lower = np.percentile(times, 5, axis=1)
        sizes_str = ["{}x{}".format(n_rows, n_cols) for n_rows, n_cols in sizes]

        color = next(colors)
        line_style = next(line_styles)

        plt.gca().set_yscale("log")
        plt.plot(sizes_str, medians, label=solver, color=color, ls=line_style)
        plt.fill_between(sizes_str, lower, upper, color=color, alpha=0.25)


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
    plt.ylabel("Time to solve (s)")
    plt.savefig(args.outputfile)


if __name__ == "__main__":
    main()
