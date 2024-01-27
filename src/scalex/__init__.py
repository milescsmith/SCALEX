# from pkg_resources import get_distribution

# __version__ = get_distribution('scalex').version
import importlib.metadata

from scalex.function import SCALEX

__version__ = importlib.metadata.version(__name__)
__author__ = importlib.metadata.metadata(__name__)["author"]
__email__ = importlib.metadata.metadata(__name__)["email"]

__all__ = ["SCALEX"]
