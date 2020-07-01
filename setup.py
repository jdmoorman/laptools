#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

with open("README.md") as readme_file:
    readme = readme_file.read()

test_requirements = [
    "codecov",
    "pytest",
    "pytest-cov",
]

docs_requirements = [
    "sphinx==1.8.5",
]

setup_requirements = [
    "pytest-runner",
    "pybind11>=2.5.0",
]

perf_requirements = [
    "pyperf",
    "matplotlib",
    "numpy",
    "scipy",
    "munkres",
    "lap",
    "lapsolver",
    "lapjv",
]

dev_requirements = [
    *test_requirements,
    *docs_requirements,
    *setup_requirements,
    *perf_requirements,
    "pre-commit",
    "bumpversion>=0.5.3",
    "ipython>=7.5.0",
    "tox>=3.5.2",
    "twine>=1.13.0",
    "wheel>=0.33.1",
]

requirements = [
    "numpy",
    "scipy",
    "numba",
]

extra_requirements = {
    "test": test_requirements,
    "docs": docs_requirements,
    "setup": setup_requirements,
    "dev": dev_requirements,
    "perf": perf_requirements,
    "all": [
        *requirements,
        *test_requirements,
        *docs_requirements,
        *setup_requirements,
        *dev_requirements,
        *perf_requirements,
    ],
}


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11

        return pybind11.get_include()


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support " "is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.define_macros = [
                ("VERSION_INFO", '"{}"'.format(self.distribution.get_version()))
            ]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    author="Jacob Moorman",
    author_email="jacob@moorman.me",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research ",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={"build_ext": BuildExt},
    description="Fast constrained linear assignment problem (CLAP) solvers",
    ext_modules=[
        Extension(
            "_augment",
            sorted(
                ["src/cpp/_augment.cpp"]
            ),  # Sort input source files to ensure bit-for-bit reproducible builds
            include_dirs=[get_pybind_include()],  # Path to pybind11 headers
            language="c++",
        ),
    ],
    extras_require=extra_requirements,
    install_requires=requirements,
    license="MIT License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="laptools",
    name="laptools",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jdmoorman/laptools",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.2.2",
    zip_safe=False,
)
