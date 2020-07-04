#include <functional>
#include <memory>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "lap.h"
#include <iostream>

static char module_docstring[] =
    "This module wraps LAPJV - Jonker-Volgenant linear sum assignment algorithm.";
static char lapjv_docstring[] =
    "Solves the linear sum assignment problem.";
static char augment_docstring[] =
    "Perform augmentation for the selected row.";

static PyObject *py_lapjv(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_augment(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef module_functions[] = {
  {"lapjv", reinterpret_cast<PyCFunction>(py_lapjv),
   METH_VARARGS | METH_KEYWORDS, lapjv_docstring},
  {"augment", reinterpret_cast<PyCFunction>(py_augment),
   METH_VARARGS | METH_KEYWORDS, augment_docstring},
  {NULL, NULL, 0, NULL}
};

extern "C" {
PyMODINIT_FUNC PyInit_py_lapjv(void) {
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "py_lapjv",          /* m_name */
      module_docstring,    /* m_doc */
      -1,                  /* m_size */
      module_functions,    /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PyModule_Create() failed");
    return NULL;
  }
  // numpy
  import_array();
  return m;
}
}

template <typename O>
using pyobj_parent = std::unique_ptr<O, std::function<void(O*)>>;

template <typename O>
class _pyobj : public pyobj_parent<O> {
 public:
  _pyobj() : pyobj_parent<O>(
      nullptr, [](O *p){ if (p) Py_DECREF(p); }) {}
  explicit _pyobj(PyObject *ptr) : pyobj_parent<O>(
      reinterpret_cast<O *>(ptr), [](O *p){ if(p) Py_DECREF(p); }) {}
  void reset(PyObject *p) noexcept {
    pyobj_parent<O>::reset(reinterpret_cast<O*>(p));
  }
};

using pyobj = _pyobj<PyObject>;
using pyarray = _pyobj<PyArrayObject>;

static PyObject *py_lapjv(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *cost_matrix_obj;
  int verbose = 0;
  int force_doubles = 0;
  static const char *kwlist[] = {
      "cost_matrix", "verbose", "force_doubles", NULL};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|pb", const_cast<char**>(kwlist),
      &cost_matrix_obj, &verbose, &force_doubles)) {
    return NULL;
  }

  // TODO: Can we accept ints?
  pyarray cost_matrix_array;
  bool float32 = true;
  cost_matrix_array.reset(PyArray_FROM_OTF(
      cost_matrix_obj, NPY_FLOAT32,
      NPY_ARRAY_IN_ARRAY | (force_doubles? 0 : NPY_ARRAY_FORCECAST)));
  if (!cost_matrix_array) {
    PyErr_Clear();
    float32 = false;
    cost_matrix_array.reset(PyArray_FROM_OTF(
        cost_matrix_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));
    if (!cost_matrix_array) {
      PyErr_SetString(PyExc_ValueError, "\"cost_matrix\" must be a numpy array "
                                        "of float32 or float64 dtype");
      return NULL;
    }
  }
  auto ndims = PyArray_NDIM(cost_matrix_array.get());
  if (ndims != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\" must be a square 2D numpy array");
    return NULL;
  }
  auto dims = PyArray_DIMS(cost_matrix_array.get());
  // if (dims[0] != dims[1]) {
  //   PyErr_SetString(PyExc_ValueError,
  //                   "\"cost_matrix\" must be a square 2D numpy array");
  //   return NULL;
  // }

  // TODO: do we check <= 0 below because of overflow in case of large arrays?
  int nr = dims[0];
  int nc = dims[1];
  // TODO: Make it work for cases when nc >= nr
  if (nr <= 0 || nc <= 0) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\"'s shape is invalid or too large");
    return NULL;
  }
  auto cost_matrix = PyArray_DATA(cost_matrix_array.get());
  npy_intp row_dims[] = {nr, 0};
  npy_intp col_dims[] = {nc, 0};
  pyarray row_ind_array(PyArray_SimpleNew(1, row_dims, NPY_INT));
  pyarray col_ind_array(PyArray_SimpleNew(1, col_dims, NPY_INT));
  auto row_ind = reinterpret_cast<int*>(PyArray_DATA(row_ind_array.get()));
  auto col_ind = reinterpret_cast<int*>(PyArray_DATA(col_ind_array.get()));
  // pyarray u_array(PyArray_SimpleNew(
  //     1, row_dims, float32? NPY_FLOAT32 : NPY_FLOAT64));
  pyarray v_array(PyArray_SimpleNew(
      1, col_dims, float32? NPY_FLOAT32 : NPY_FLOAT64));
  // double lapcost;
  if (float32) {
    // auto u = reinterpret_cast<float*>(PyArray_DATA(u_array.get()));
    auto v = reinterpret_cast<float*>(PyArray_DATA(v_array.get()));
    Py_BEGIN_ALLOW_THREADS
    lap(nr, nc, reinterpret_cast<float*>(cost_matrix),
        row_ind, col_ind, v, verbose);
    Py_END_ALLOW_THREADS
  } else {
    // auto u = reinterpret_cast<double*>(PyArray_DATA(u_array.get()));
    auto v = reinterpret_cast<double*>(PyArray_DATA(v_array.get()));
    Py_BEGIN_ALLOW_THREADS
    lap(nr, nc, reinterpret_cast<double*>(cost_matrix),
        row_ind, col_ind, v, verbose);
    Py_END_ALLOW_THREADS
  }
  return Py_BuildValue("(OOO)",
                       row_ind_array.get(), col_ind_array.get(), v_array.get());
}


// TODO: Add an augment function that interacts with outside python objects
static PyObject *py_augment(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *cost_matrix_obj;
  PyObject *col4row_obj, *row4col_obj, *v_obj;
  int freerow = 0;
  int verbose = 0;
  int force_doubles = 0;
  static const char *kwlist[] = {
      "cost_matrix", "freerow", "col4row", "row4col", "v",
      "verbose", "force_doubles", NULL};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "ObOOO|pb", const_cast<char**>(kwlist),
      &cost_matrix_obj, &freerow, &col4row_obj, &row4col_obj, &v_obj,
      &verbose, &force_doubles)) {
    return NULL;
  }
  pyarray cost_matrix_array;
  bool float32 = true;
  cost_matrix_array.reset(PyArray_FROM_OTF(
      cost_matrix_obj, NPY_FLOAT32,
      NPY_ARRAY_IN_ARRAY | (force_doubles? 0 : NPY_ARRAY_FORCECAST)));
  if (!cost_matrix_array) {
    PyErr_Clear();
    float32 = false;
    cost_matrix_array.reset(PyArray_FROM_OTF(
        cost_matrix_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));
    if (!cost_matrix_array) {
      PyErr_SetString(PyExc_ValueError, "\"cost_matrix\" must be a numpy array "
                                        "of float32 or float64 dtype");
      return NULL;
    }
  }

  if (float32){
    std::cout << "The datatype of the cost matrix is float32." << std::endl;
  } else {
    std::cout << "The datatype of the cost matrix is float64." << std::endl;
  }

  pyarray col4row_array;
  col4row_array.reset(PyArray_FROM_OTF(
      col4row_obj, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
  if (!col4row_array) {
    // TODO: Do we need to set the python error string here?
    // PyErr_SetString(PyExc_ValueError, "\"col4row\" must be a numpy array "
    //                                   "of int dtype");
    return NULL;
  }

  pyarray row4col_array;
  row4col_array.reset(PyArray_FROM_OTF(
      row4col_obj, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
  if (!row4col_array) {
    // TODO: Do we need to set the python error string here?
    // PyErr_SetString(PyExc_ValueError, "\"row4col\" must be a numpy array "
    //                                   "of int dtype");
    return NULL;
  }

  // TODO: Verify that u is never copied but rather modified in place.
  // pyarray u_array;
  // if (float32) {
  //   u_array.reset(PyArray_FROM_OTF(
  //       u_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
  // } else {
  //   u_array.reset(PyArray_FROM_OTF(
  //       u_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
  // }
  // if (!u_array) {
  //   PyErr_SetString(PyExc_ValueError, "\"u\" must be a numpy array "
  //                                     "of float32 or float64 dtype");
  //   return NULL;
  // }

  // TODO: Verify that v is never copied but rather modified inplace.
  pyarray v_array;
  if (float32) {
    v_array.reset(PyArray_FROM_OTF(
        v_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
  } else {
    v_array.reset(PyArray_FROM_OTF(
        v_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
  }
  if (!v_array) {
    PyErr_SetString(PyExc_ValueError, "\"v\" must be a numpy array "
                                      "of float32 or float64 dtype");
    return NULL;
  }

  auto ndims = PyArray_NDIM(cost_matrix_array.get());
  if (ndims != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\" must be a square 2D numpy array");
    return NULL;
  }
  auto dims = PyArray_DIMS(cost_matrix_array.get());
  int nr = dims[0];
  int nc = dims[1];
  if (nr <= 0 || nc <= 0) {
    PyErr_SetString(PyExc_ValueError, "\"cost_matrix\"'s shape is invalid");
    return NULL;
  }
  auto cost_matrix = PyArray_DATA(cost_matrix_array.get());
  auto col4row = reinterpret_cast<int*>(PyArray_DATA(col4row_array.get()));
  auto row4col = reinterpret_cast<int*>(PyArray_DATA(row4col_array.get()));
  // auto u = PyArray_DATA(u_array.get());
  auto v = PyArray_DATA(v_array.get());

  if (float32) {
    Py_BEGIN_ALLOW_THREADS
    augment(freerow, nr, nc,
            reinterpret_cast<float*>(cost_matrix),
            col4row,
            row4col,
            reinterpret_cast<float*>(v),
            verbose);
    Py_END_ALLOW_THREADS
  } else {
    Py_BEGIN_ALLOW_THREADS
    augment(freerow, nr, nc,
            reinterpret_cast<double*>(cost_matrix),
            col4row,
            row4col,
            reinterpret_cast<double*>(v),
            verbose);
    Py_END_ALLOW_THREADS
  }

  for (int i = 0; i < nr; i++){
    std::cout << col4row[i] << std::endl;
  }

  for (int i = 0; i < nc; i++){
    std::cout << row4col[i] << std::endl;
  }

  return Py_BuildValue("(OOO)",
                       col4row_array.get(), row4col_array.get(), v_array.get());
}
