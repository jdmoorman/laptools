#include <pybind11/pybind11.h>

int augment(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(_augment, m) {
    m.def("augment", &augment, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
}
