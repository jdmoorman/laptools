![Black Logo](logo.png)

<h2 align="center">Fast Assignment Problem Solvers</h2>

<p align="center">
<a href="https://pypi.org/project/laptools/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/laptools.svg"></a>
<a href="https://pypi.org/project/laptools/"><img alt="Supported Python Versions" src="https://img.shields.io/pypi/pyversions/laptools.svg"></a>
<a href="https://github.com/jdmoorman/laptools/actions"><img alt="Build Status" src="https://github.com/jdmoorman/laptools/workflows/CI/badge.svg"></a>
<a href="https://codecov.io/gh/jdmoorman/laptools"><img alt="Code Coverage" src="https://codecov.io/gh/jdmoorman/laptools/branch/master/graph/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

---

## Installation
**Stable Release:** `pip install laptools`<br>
**Development Head:** `pip install git+https://github.com/jdmoorman/laptools.git`

## Quick Start

Import the the `clap` module from the package and define your cost matrix.

```python
>>> from laptools import clap
>>> costs = [[0, 0, 1],
...          [1, 0, 2]]

```

Solve the linear assignment problem with row 0 forcibly assigned to column 1.

```python
>>> clap.cost(0, 1, costs)
1

```

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
