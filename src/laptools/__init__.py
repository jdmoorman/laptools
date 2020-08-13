"""Top-level package for laptools."""

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.2.6"

__author__ = "Jacob Moorman"
__email__ = "jacob@moorman.me"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2020, Jacob Moorman"

from _augment import _solve
from py_lapjv import augment, lapjv

from . import clap, clap_naive, lap

__all__ = ["clap", "clap_naive", "lap", "_solve", "lapjv", "augment"]
