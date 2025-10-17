from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from scalex.function import SCALEX, label_transfer

try:
    if __package__ is not None:
        __version__ = version(__package__)
    else:
        __version__ = "unknown"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

logger.disable(__package__)

__all__ = ["SCALEX", "label_transfer"]