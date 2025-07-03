"""
Logging configuration for the transcriber.
"""

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(
    level: str = "INFO", 
    log_file: Optional[Path] = None,
    debug: bool = False
) -> logging.Logger:
    """
    Set up logging with Rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        debug: Enable debug mode with more verbose output
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("transcriber")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create Rich handler for console output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=debug,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(getattr(logging, level.upper()))
    
    # Format for Rich handler
    if debug:
        rich_format = "%(name)s: %(message)s"
    else:
        rich_format = "%(message)s"
    
    rich_handler.setFormatter(logging.Formatter(rich_format))
    logger.addHandler(rich_handler)
    
    # Add file handler if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)
    
    return logger