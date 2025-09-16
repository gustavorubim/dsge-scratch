"""SGE Labs - From-scratch DSGE toolkit."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sgelabs")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
