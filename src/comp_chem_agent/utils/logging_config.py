import logging
import sys


def setup_logger(name=None, level=logging.INFO):
    """
    Set up a logger with consistent formatting.

    Args:
        name (str, optional): Logger name. If None, returns root logger
        level (int, optional): Logging level. Defaults to INFO

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # Only add handler if none exists
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
