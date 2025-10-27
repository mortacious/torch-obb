"""torch-obb: High-performance oriented bounding box intersection operations for PyTorch."""

from .intersection import obb_intersection_volumes, obb_overlaps, obb_ious

# Import version from setuptools-scm generated file
try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    # Fallback for development/installation when _version.py doesn't exist yet
    __version__ = "0.0.0.dev0"

__author__ = "Felix Igelbrink"
__email__ = "felix.igelbrink@gmail.com"
__license__ = "MIT"
__all__ = ["obb_intersection_volumes", "obb_overlaps", "obb_ious", "__version__", "__author__", "__email__", "__license__"]