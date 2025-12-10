"""Package initializer for src.

This file ensures `src` is treated as a package so tools like mypy
and Python imports resolve consistently.
"""

# Expose commonly used module names here if needed
from . import config, models, train, utils

__all__ = ["config", "models", "train", "utils"]
