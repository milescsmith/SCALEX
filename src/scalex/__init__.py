# from pkg_resources import get_distribution

# __version__ = get_distribution('scalex').version
from importlib.metadata import PackageNotFoundError, metadata, version

from loguru import logger

from scalex.function import SCALEX

try:
    __version__ = version(__name__)
    __author__ = metadata(__name__)["author"]
    __email__ = metadata(__name__)["email"]
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

logger.disable("scalex")

__all__ = ["SCALEX"]
