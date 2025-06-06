from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from scalex.function import SCALEX, label_transfer

try:
    __version__ = version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

logger.disable(__package__)

__all__ = ["SCALEX", "label_transfer"]
