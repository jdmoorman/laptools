# clapsolver

[![Build Status](https://github.com/jdmoorman/clapsolver/workflows/Build%20Master/badge.svg)](https://github.com/jdmoorman/clapsolver/actions)
[![Documentation](https://readthedocs.org/projects/clapsolver/badge/?version=latest)](https://clapsolver.readthedocs.io/en/latest/?badge=latest)
[![Code Coverage](https://codecov.io/gh/jdmoorman/clapsolver/branch/master/graph/badge.svg)](https://codecov.io/gh/jdmoorman/clapsolver)

Fast constrained linear assignment problem (CLAP) solvers for Python

---

## Features
* Store values and retain the prior value in memory
* ... some other functionality

## Quick Start
```python
>>> from clapsolver import Example
>>> a = Example()
>>> a.get_value()
10

```

## Installation
**Stable Release:** `pip install clapsolver`<br>
**Development Head:** `pip install git+https://github.com/jdmoorman/clapsolver.git`

## Documentation
TODO: readthedocs
For full package documentation please visit [jdmoorman.github.io/clapsolver](https://jdmoorman.github.io/clapsolver).

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## The Commands You Need To Know
1. `pip install -e .[dev]`

    This will install your package in editable mode with all the required development dependencies (i.e. `tox`).

2. `tox -e <env>`

    This will use tox to run the steps outlined in the `[testenv:<env>]` section of `tox.ini`.

#### Additional Optional Setup Steps:
* Make sure the github repository initialized correctly at
    * `https://github.com:jdmoorman/clapsolver.git`
* Register clapsolver with Codecov:
  * Make an account on [codecov.io](https://codecov.io) (Recommended to sign in with GitHub)
  * Select `jdmoorman` and click: `Add new repository`
  * Copy the token provided, go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/jdmoorman/clapsolver/settings/secrets),
  add a secret called `CODECOV_TOKEN` with the token you just copied.
  Don't worry, no one will see this token because it will be encrypted.
* Register your project with PyPI:
  * Make an account on [pypi.org](https://pypi.org)
  * Go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/jdmoorman/clapsolver/settings/secrets),
  add a secret called `PYPI_TOKEN` with your password for your PyPI account.
  Don't worry, no one will see this password because it will be encrypted.
  * Next time you push to the branch: `stable`, GitHub actions will build and deploy your Python package to PyPI.
  * _Recommendation: Prior to pushing to `stable` it is recommended to install and run `bumpversion` as this will,
  tag a git commit for release and update the `setup.py` version number._
* Add branch protections to `master` and `stable`
    * To protect from just anyone pushing to `master` or `stable` (the branches with more tests and deploy
    configurations)
    * Go to your [GitHub repository's settings and under the `Branches` tab](https://github.com/jdmoorman/clapsolver/settings/branches), click `Add rule` and select the
    settings you believe best.
    * _Recommendations:_
      * _Require pull request reviews before merging_
      * _Require status checks to pass before merging (Recommended: lint and test)_

#### Suggested Git Branch Strategy
1. `master` is for the most up-to-date development, very rarely should you directly commit to this branch. GitHub
Actions will run on every push and on a CRON to this branch but still recommended to commit to your development
branches and make pull requests to master.
2. `stable` is for releases only. When you want to release your project on PyPI, simply make a PR from `master` to
`stable`, this template will handle the rest as long as you have added your PyPI information described in the above
**Optional Steps** section.
3. Your day-to-day work should exist on branches separate from `master`. Even if it is just yourself working on the
repository, make a PR from your working branch to `master` so that you can ensure your commits don't break the
development head. GitHub Actions will run on every push to any branch or any pull request from any branch to any other
branch.
4. It is recommended to use "Squash and Merge" commits when committing PR's. It makes each set of changes to `master`
atomic and as a side effect naturally encourages small well defined PR's.
5. GitHub's UI is bad for rebasing `master` onto `stable`, as it simply adds the commits to the other branch instead of
properly rebasing from what I can tell. You should always rebase locally on the CLI until they fix it.
