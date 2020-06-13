"""Top-level package for laptools."""

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.8"

__author__ = "Jacob Moorman"
__email__ = "jacob@moorman.me"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2020, Jacob Moorman"


from _augment import _solve, augment

from . import clap, clap_new, lap, lap_cpp

__all__ = ["clap", "clap_new", "lap", "lap_cpp", "augment", "_solve"]
