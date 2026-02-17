from loguru import logger
import sys
import os


def setup_logging(service_name: str = "app", log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}"
    )

    logger.add(
        f"{log_dir}/{service_name}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.add(
        f"{log_dir}/{service_name}_errors.log",
        rotation="10 MB",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return logger
