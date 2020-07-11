/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
This code implements the shortest augmenting path algorithm for the
rectangular assignment problem.  This implementation is based on the
pseudocode described in pages 1685-1686 of:
    DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952
Author: PM Larsen
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>

// TODO: If moving the allocations outside of augment makes things faster,
// we may need to have separate implementations of augment for c++ vs. python.
template <class TIndex, class TCost>
void
augment(py::array_t<TCost> cost_matrix,
        TIndex cur_row,
        py::array_t<TIndex> row4col,
        py::array_t<TIndex> col4row,
        py::array_t<TCost> u,
        py::array_t<TCost> v)
{
    // u is a numpy array, we don't know how to access its data
    auto cost_data = cost_matrix.template unchecked<2>();
    auto u_data = u.template mutable_unchecked<1>();
    auto v_data = v.template mutable_unchecked<1>();
    auto row4col_data = row4col.template mutable_unchecked<1>();
    auto col4row_data = col4row.template mutable_unchecked<1>();

    // u_data is the data from the numpy array u. i.e. u(i) is the i'th element.

    TCost minVal = 0;
    TIndex row_idx = cur_row;
    TIndex nr = cost_matrix.shape(0);
    TIndex nc = cost_matrix.shape(1);

    // Crouse's pseudocode uses set complements to keep track of remaining
    // nodes.  Here we use a vector, as it is more efficient in C++.
    TIndex num_remaining = nc;
    // TODO: Try moving the allocation outside the augment function to be done just once.
    std::vector<TIndex> remaining(nc);
    for (TIndex it = 0; it < nc; it++) {
        // Filling this up in reverse order ensures that the solution of a
        // constant cost matrix is the identity matrix (c.f. #11602).
        // remaining[it] = nc - it - 1;
        remaining[it] = it;
    }

    // TODO: Try moving the allocation outside the augment function to be done just once.
    std::vector<TIndex> path(nc, -1);
    std::vector<TCost> shortestPathCosts(nc);
    std::fill(shortestPathCosts.begin(), shortestPathCosts.end(), INFINITY);

    // TODO: Try moving the allocation outside the augment function to be done just once.
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);
    std::fill(SR.begin(), SR.end(), false);
    std::fill(SC.begin(), SC.end(), false);

    // find shortest augmenting path
    TIndex sink = -1;
    while (sink == -1) {

        TIndex index = -1;
        TCost lowest = INFINITY;
        SR[row_idx] = true;

        for (TIndex it = 0; it < num_remaining; it++) {
            TIndex j = remaining[it];

            TCost r = minVal + cost_data(row_idx, j)- u_data(row_idx) - v_data(j);
            if (r < shortestPathCosts[j]) {
                path[j] = row_idx;
                shortestPathCosts[j] = r;
            }

            // When multiple nodes have the minimum cost, we select one which
            // gives us a new sink node. This is particularly important for
            // integer cost matrices with small co-efficients.
            if (shortestPathCosts[j] < lowest ||
                (shortestPathCosts[j] == lowest && row4col_data(j) == -1)) {
                lowest = shortestPathCosts[j];
                index = it;
            }
        }

        minVal = lowest;
        TIndex j = remaining[index];

        // TODO: raise an exception if minVal is INFINITY
        if (minVal == INFINITY) { // infeasible cost matrix
            throw pybind11::value_error("cost matrix is infeasible");
        }

        if (row4col_data(j) == -1) {
            sink = j;
        } else {
            row_idx = row4col_data(j);
        }

        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
        remaining.resize(num_remaining);
    }

    // update dual variables
    // u_data(row_idx) += minVal;
    for (TIndex i = 0; i < nr; i++) {
        if (SR[i]) {
            if (i == cur_row) {
                u_data(i) += minVal;
            }
            else {
                u_data(i) += minVal - shortestPathCosts[col4row_data(i)];
            }
        }
    }

    for (TIndex j = 0; j < nc; j++) {
        if (SC[j]) {
            v_data(j) -= minVal - shortestPathCosts[j];
        }
    }

    // augment previous solution
    TIndex col_idx = sink;
    while (1) {
        row_idx = path[col_idx];
        row4col_data(col_idx) = row_idx;
        std::swap(col4row_data(row_idx), col_idx);
        if (row_idx == cur_row) {
            break;
        }
    }
}

template <class T>
py::array_t<T>  _fill(py::array_t<T> arr, T val) {
    std::fill(arr.mutable_data(), arr.mutable_data() + arr.size(), val);
    return arr;
}

template <class T>
py::tuple
_solve(py::array_t<T> cost_matrix)
{
    long nr = cost_matrix.shape(0);
    long nc = cost_matrix.shape(1);

    py::array_t<T> u = py::array_t<T>(nr);
    py::array_t<T> v = py::array_t<T>(nc);
    py::array_t<long> row4col = py::array_t<long>(nc);
    py::array_t<long> col4row = py::array_t<long>(nr);

    // TODO: Can we skip this or fold it into the allocation step above?
    _fill<T>(u, 0.);
    _fill<T>(v, 0.);
    _fill<long>(row4col, -1);
    _fill<long>(col4row, -1);

    // TODO: We only use cost_matrix through cost_matrix.unchecked<2>()
    for (long cur_row = 0; cur_row < nr; cur_row++) {
        augment<long, T>(cost_matrix, cur_row, row4col, col4row, u, v);
    }

    return py::make_tuple(row4col, col4row, u, v);

}

template <class TIndex, class TCost>
void def_augment(py::module m) {
    m.def("augment", &augment<TIndex, TCost>,
        R"pbdoc(
            TODO: Docstring.
        )pbdoc",
        py::arg("cost_matrix"),
        py::arg("cur_row").noconvert(),
        py::arg("row4col").noconvert(),
        py::arg("col4row").noconvert(),
        py::arg("u").noconvert(),
        py::arg("v").noconvert()
    );
}

template <class T>
void def_solve(py::module m) {
    m.def("_solve", &_solve<T>, R"pbdoc(
        Solve the linear assignment problem.

        TODO: Docstring.
    )pbdoc");
    // TODO: noconvert
}

PYBIND11_MODULE(_augment, m) {
    def_augment<long, double>(m);
    def_solve<double>(m);
}
