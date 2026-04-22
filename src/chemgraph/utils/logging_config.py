import logging
import sys
import warnings

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


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
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(_LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    # Prevent double logging when the root logger is also configured by callers (e.g., Streamlit).
    logger.propagate = False
    return logger


def configure_logging(level: int = logging.WARNING) -> None:
    """Set the log level for all ``chemgraph.*`` loggers.

    Call this once early in the CLI entry point to control verbosity
    for the entire package.  The level applies to the ``"chemgraph"``
    namespace logger and is propagated to every already-created child
    logger (e.g. ``chemgraph.models.openai``,
    ``chemgraph.graphs.single_agent``).

    Parameters
    ----------
    level : int
        A :mod:`logging` level constant (e.g. ``logging.WARNING``,
        ``logging.INFO``, ``logging.DEBUG``).
    """
    # Configure the root "chemgraph" namespace logger.
    root = logging.getLogger("chemgraph")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(_LOG_FORMAT)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Propagate the level to any already-created child loggers so that
    # modules imported before this call also respect the new level.
    manager = logging.Logger.manager
    for name, logger_ref in manager.loggerDict.items():
        if isinstance(logger_ref, logging.Logger) and name.startswith("chemgraph."):
            logger_ref.setLevel(level)
            for handler in logger_ref.handlers:
                handler.setLevel(level)

    # Suppress noisy third-party warnings when not in verbose mode.
    if level > logging.INFO:
        warnings.filterwarnings("ignore", category=UserWarning, module=r"langchain.*")
    else:
        # Re-enable if user asks for verbose output.
        warnings.filterwarnings("default", category=UserWarning, module=r"langchain.*")
