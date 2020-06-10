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

// // Note that though this function expects arrays of doubles, it will accept
// // arrays of any type and cast them silently. Thus, for best performance,
// // be sure to convert arrays to doubles in advance.
// double augment(py::array_t<double> cost_matrix) {
//     auto cost_data = cost_matrix.unchecked<2>();
//     double sum = 0;
//     for (auto i = 0; i < cost_data.shape(0); i++) {
//         for (auto j = 0; j < cost_data.shape(1); j++) {
//             sum += cost_data(i, j);
//         }
//     }
//     return sum;
// }


#include <algorithm>
#include <cmath>
#include <vector>


void
augment(py::array_t<double> cost_matrix,
        int cur_row,
        py::array_t<int> row4col,
        py::array_t<int> col4row,
        py::array_t<double> u,
        py::array_t<double> v)
{
    // u is a numpy array, we don't know how to access its data
    auto cost_data = cost_matrix.unchecked<2>();
    auto u_data = u.mutable_unchecked<1>();
    auto v_data = v.mutable_unchecked<1>();
    auto row4col_data = row4col.mutable_unchecked<1>();
    auto col4row_data = col4row.mutable_unchecked<1>();

    // u_data is the data from the numpy array u. i.e. u(i) is the i'th element.

    int row_idx = cur_row;
    double minVal = 0;
    int nr = cost_matrix.shape(0);
    int nc = cost_matrix.shape(1);

    // Crouse's pseudocode uses set complements to keep track of remaining
    // nodes.  Here we use a vector, as it is more efficient in C++.
    int num_remaining = nc;
    std::vector<int> remaining(nc);
    for (int it = 0; it < nc; it++) {
        // Filling this up in reverse order ensures that the solution of a
        // constant cost matrix is the identity matrix (c.f. #11602).
        remaining[it] = nc - it - 1;
    }

    std::vector<int> path(nc, -1);
    std::vector<double> shortestPathCosts(nc);
    std::fill(shortestPathCosts.begin(), shortestPathCosts.end(), INFINITY);

    // TODO: Decide whether to take the allocation outside the augment function.
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);
    std::fill(SR.begin(), SR.end(), false);
    std::fill(SC.begin(), SC.end(), false);

    // find shortest augmenting path
    int sink = -1;
    while (sink == -1) {
        py::print("cur_row start", row_idx);

        std::cout << "remaining" << " ";
        for(auto itr : remaining)
            std::cout << itr << " ";
        std::cout << std::endl;

        int index = -1;
        double lowest = INFINITY;
        SR[row_idx] = true;

        for (int it = 0; it < num_remaining; it++) {
            int j = remaining[it];

            double r = minVal + cost_data(row_idx, j)- u_data(row_idx) - v_data(j);
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

        std::cout << "path" << " ";
        for(auto itr : path)
            std::cout << itr << " ";
        std::cout << std::endl;

        std::cout << "shortest path costs" << " ";
        for(auto itr : shortestPathCosts)
            std::cout << itr << " ";
        std::cout << std::endl;

        int j = remaining[index];

        py::print("idx and min", j, lowest);

        minVal = lowest;

        // TODO: raise an exception if minVal is INFINITY
        // if (minVal == INFINITY) { // infeasible cost matrix
        //     return -1;
        // }

        if (row4col_data(j) == -1) {
            sink = j;
        } else {
            row_idx = row4col_data(j);
        }

        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
        remaining.resize(num_remaining);

        py::print("cur_row end", row_idx);
    }


    // update dual variables
    // u_data(row_idx) += minVal;
    for (int i = 0; i < nr; i++) {
        if (SR[i]) {
            py::print("i", i);
            py::print("cur_row", cur_row);
            if (i == row_idx) {
                u_data(i) += minVal;
            }
            else {
                py::print("col4row(i)", col4row_data(i));
                u_data(i) += minVal - shortestPathCosts[col4row_data(i)];
            }
        }
    }
    py::print("u", u);

    for (int j = 0; j < nc; j++) {
        if (SC[j]) {
            v_data(j) -= minVal - shortestPathCosts[j];
        }
    }
    py::print("v", v);

    // augment previous solution
    int col_idx = sink;
    while (1) {
        row_idx = path[col_idx];
        row4col_data(col_idx) = row_idx;
        std::swap(col4row_data(row_idx), col_idx);
        if (row_idx == cur_row) {
            break;
        }
        py::print("Stuck?");
    }

    py::print("assignments", row4col, col4row);
}

template <class T>
py::array_t<T>  _fill(py::array_t<T> arr, T val) {
    std::fill(arr.mutable_data(), arr.mutable_data() + arr.size(), val);
    return arr;
}

// TODO: Figure out how to do templates: template <class T>, py::array_t<T>
py::tuple
_solve(py::array_t<double> cost_matrix)
{
    int nr = cost_matrix.shape(0);
    int nc = cost_matrix.shape(1);

    py::array_t<double> u = py::array_t<double>(nr);
    py::array_t<double> v = py::array_t<double>(nc);
    py::array_t<int> row4col = py::array_t<int>(nc);
    py::array_t<int> col4row = py::array_t<int>(nr);

    // TODO: Can we skip this or fold it into the allocation step above?
    _fill<double>(u, 0.);
    _fill<double>(v, 0.);
    _fill<int>(row4col, -1);
    _fill<int>(col4row, -1);

    // TODO: We only use cost_matrix through cost_matrix.unchecked<2>()
    for (int cur_row = 0; cur_row < nr; cur_row++) {
        py::print("Current row:", cur_row);
        augment(cost_matrix, cur_row, row4col, col4row, u, v);
    }

    return py::make_tuple(row4col, col4row, u, v);
    // int nr = cost_matrix.shape(0);
    // int nc = cost_matrix.shape(1);
    //
    // // initialize variables
    // std::vector<double> u(nr, 0);
    // std::vector<double> v(nc, 0);
    // std::vector<double> shortestPathCosts(nc);
    // std::vector<int> path(nc, -1);
    // std::vector<int> col4row(nr, -1);
    // std::vector<int> row4col(nc, -1);
    // std::vector<bool> SR(nr);
    // std::vector<bool> SC(nc);
    //
    // // iteratively build the solution
    // for (int cur_row = 0; cur_row < nr; cur_row++) {
    //
    //     double minVal;
    //     int sink = augmenting_path(nc, cost, u, v, path, row4col,
    //                                shortestPathCosts, curRow, SR, SC, &minVal);
    //     if (sink < 0) {
    //         return -1;
    //     }
    //
    //     // update dual variables
    //     u[curRow] += minVal;
    //     for (int i = 0; i < nr; i++) {
    //         if (SR[i] && i != curRow) {
    //             u[i] += minVal - shortestPathCosts[col4row[i]];
    //         }
    //     }
    //
    //     for (int j = 0; j < nc; j++) {
    //         if (SC[j]) {
    //             v[j] -= minVal - shortestPathCosts[j];
    //         }
    //     }
    //
    //     // augment previous solution
    //     int j = sink;
    //     while (1) {
    //         int i = path[j];
    //         row4col[j] = i;
    //         std::swap(col4row[i], j);
    //         if (i == curRow) {
    //             break;
    //         }
    //     }
    // }
    //
    // for (int i = 0; i < nr; i++) {
    //     output_col4row[i] = col4row[i];
    // }
    //
    // return 0;

    //
    // std::vector<int> col4row(nr, -1);
    // std::vector<int> row4col(nc, -1);
    //
    //
    // std::vector<double> u(nr, 0);
    // std::vector<double> v(nc, 0);
    //
    // return py::make_tuple(ret);
}

PYBIND11_MODULE(_augment, m) {
    m.def("augment", &augment, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
    m.def("_solve", &_solve, R"pbdoc(
        Solve the linear assignment problem.
    )pbdoc");
}
