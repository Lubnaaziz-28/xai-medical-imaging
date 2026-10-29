"""Logger setup for MedCard."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging for MedCard."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger("medcard")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the MedCard logger."""
    return logging.getLogger("medcard")