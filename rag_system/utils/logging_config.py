"""
Logging configuration for the RAG system.

Call setup_logging() once at the start of each service entry-point.
"""
import sys
import os
from loguru import logger


def setup_logging(
    service_name: str = "app",
    log_dir: str = "logs",
    level: str = "INFO",
    json_logs: bool = False,
) -> "logger":
    """
    Configure loguru for a service.

    Args:
        service_name: Used as the log filename prefix.
        log_dir:      Directory where log files are written.
        level:        Minimum log level for the console sink.
        json_logs:    If True, file sinks emit newline-delimited JSON.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()

    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}"

    logger.add(sys.stderr, level=level.upper(), format=fmt, colorize=True)

    logger.add(
        f"{log_dir}/{service_name}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        serialize=json_logs,
        enqueue=True,
    )

    logger.add(
        f"{log_dir}/{service_name}_errors.log",
        rotation="10 MB",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        serialize=json_logs,
        enqueue=True,
    )

    return logger
