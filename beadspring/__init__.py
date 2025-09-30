"""Top-level namespace for the :mod:`beadspring` package.

This module exposes the two high-level subpackages :mod:`beadspring.analysis`
and :mod:`beadspring.utils` while keeping their respective namespaces tidy.
Importing everything into the top-level namespace via ``import *`` made it
harder to discover where a particular function lived and regularly triggered
``flake8`` warnings in downstream projects.  The new explicit exports keep the
public API intact while encouraging ``beadspring.analysis`` and
``beadspring.utils`` to be accessed through their dedicated modules.
"""

from . import analysis, utils

__all__ = ["analysis", "utils"]

