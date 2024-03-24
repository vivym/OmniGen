import os
import sys
import logging
import threading

_lock = threading.Lock()
_default_handler: logging.Handler | None = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_default_logging_level() -> int:
    """
    If OMNI_GEN_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("OMNI_GEN_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option OMNI_GEN_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _configure_default_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return

        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = logging.getLogger(__name__.split(".")[0])
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Get logger with given name.

    Args:
        name (str, optional): Name of logger. Defaults to None.

    Returns:
        logging.Logger: Logger.
    """
    if name is None:
        name = __name__.split(".")[0]

    _configure_default_root_logger()
    return logging.getLogger(name)
