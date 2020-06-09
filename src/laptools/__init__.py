"""Top-level package for laptools."""

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.8"

__author__ = "Jacob Moorman"
__email__ = "jacob@moorman.me"

__license__ = "MIT license".rstrip(" license")
__copyright__ = "Copyright (c) 2020, Jacob Moorman"


from _augment import augment

from . import clap, clap_new, lap

__all__ = ["clap", "lap", "augment"]
