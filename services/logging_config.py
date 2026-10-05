import logging
import sys


def configure_logging():
    """
    Configure application-wide structured logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def get_logger(name: str):
    """
    Return a logger for the requested module.
    """

    return logging.getLogger(name)