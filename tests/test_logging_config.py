import io
import logging

import pytest

from chemgraph.utils.logging_config import configure_logging


@pytest.fixture
def isolated_chemgraph_logging():
    """Isolate global logger mutations made by ``configure_logging``."""
    root = logging.getLogger()
    package = logging.getLogger("chemgraph")
    child = logging.getLogger("chemgraph.tools.logging_test")
    loggers = {root, package, child}
    loggers.update(
        logger
        for name, logger in logging.Logger.manager.loggerDict.items()
        if name.startswith("chemgraph.") and isinstance(logger, logging.Logger)
    )
    snapshots = {
        logger: (
            logger.level,
            logger.propagate,
            list(logger.handlers),
            {handler: handler.level for handler in logger.handlers},
        )
        for logger in loggers
    }

    root.handlers = []
    package.handlers = []
    child.handlers = []
    root.setLevel(logging.DEBUG)
    package.setLevel(logging.NOTSET)
    child.setLevel(logging.NOTSET)
    package.propagate = True
    child.propagate = True

    try:
        yield root, package, child
    finally:
        for logger, (level, propagate, handlers, handler_levels) in snapshots.items():
            for handler in logger.handlers:
                if handler not in handlers:
                    handler.close()
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate
            for handler, handler_level in handler_levels.items():
                handler.setLevel(handler_level)


def test_configure_logging_stops_root_duplication_and_keeps_child_file_log(
    isolated_chemgraph_logging,
    tmp_path,
):
    root, package, child = isolated_chemgraph_logging
    root_output = io.StringIO()
    terminal_output = io.StringIO()
    root.addHandler(logging.StreamHandler(root_output))
    file_handler = logging.FileHandler(tmp_path / "ase_core.log")
    child.addHandler(file_handler)

    configure_logging(logging.INFO)
    configure_logging(logging.INFO)
    assert len(package.handlers) == 1
    package.handlers[0].setStream(terminal_output)

    child.info("simulation completed")
    file_handler.flush()

    assert package.propagate is False
    assert terminal_output.getvalue().count("simulation completed") == 1
    assert root_output.getvalue() == ""
    assert (tmp_path / "ase_core.log").read_text().count("simulation completed") == 1
