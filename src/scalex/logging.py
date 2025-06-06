import datetime
from sys import stderr

from loguru import logger


def init_logger(verbose: int, save_log: bool = True, msg_format: str | None = None) -> None:
    logger.enable(__package__)
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo

    try:
        from IPython import get_ipython

        in_notebook = get_ipython() is not None
    except ImportError:
        in_notebook = False

    if msg_format is None:
        if in_notebook:
            msg_format = (
                "{level}|<green>{name}</green>:<red>{function}</red>:<blue>{line}</blue> - <level>{message}</level>"
            )
        else:
            msg_format = "{time:YYYY-MM-DD HH:mm:ss}|{level}|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    match verbose:
        case 3:
            log_level = "DEBUG"  # catches DEBUG, INFO, WARNING, ERROR, CRITICAL, and EXCEPTION
        case 2:
            log_level = "INFO"  # catches INFO, WARNING, ERROR, CRITICAL, and EXCEPTION
        case 1:
            log_level = "WARNING"  # catches WARNING, ERROR, CRITICAL, and EXCEPTION
        case _:
            log_level = "ERROR"  # Catches ERROR, CRITICAL, and EXCEPTION

    config = {"handlers": [{"sink": stderr, "format": msg_format, "level": log_level}]}
    logger.configure(**config)

    if save_log:
        logger.add(
            sink=f"{__package__}_{datetime.datetime.now(tz=timezone).strftime('%Y-%d-%m--%H-%M-%S')}.log",
            level=log_level,
            format="{time:YYYY-MM-DD at HH:mm:ss}|{level}|{name}:{function}:{line} - {message}",
            filter=__package__,
            backtrace=True,
            diagnose=True,
            colorize=False,
        )
