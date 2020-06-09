#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double augment(py::array_t<double> cost_matrix) {
    auto cost_data = cost_matrix.unchecked<2>();
    double sum = 0;
    for (size_t i = 0; i < cost_data.shape(0); i++) {
        for (size_t j = 0; j < cost_data.shape(1); j++) {
            sum += cost_data(i, j);
        }
    }
    return sum;
}

PYBIND11_MODULE(_augment, m) {
    m.def("augment", &augment, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
}
