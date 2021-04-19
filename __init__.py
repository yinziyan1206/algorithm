"""
    NaYuan Local Algorithm Utility
"""
from __future__ import absolute_import
# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

__author__ = "ziyan.yin(yinziyan@nayuan.net"
__version__ = "0.0.1"


logging.getLogger(__name__).addHandler(NullHandler())


def add_stderr_logger(level=logging.DEBUG):
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging.

    Returns the handler after adding it.
    """
    # This method needs to be in this __init__.py to get the __name__ correct
    # even if urllib3 is vendored within another package.
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


# ... Clean up.
del NullHandler
