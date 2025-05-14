import logging
import sys


def setup_logger(name=None, level=logging.INFO):
    """Set up a logger with consistent formatting.

    This function configures a logger with a standard format that includes
    timestamp, logger name, log level, and message. It ensures that handlers
    are not duplicated if the logger already exists.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns the root logger, by default None
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG), by default logging.INFO

    Returns
    -------
    logging.Logger
        Configured logger instance with the specified name and level

    Notes
    -----
    The logger format includes:
    - Timestamp
    - Logger name
    - Log level
    - Message
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # Only add handler if none exists
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
