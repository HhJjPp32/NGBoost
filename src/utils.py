"""
Utility functions for logging, configuration loading, and helper operations.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        format_str: Optional custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_str)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save the YAML file
    """
    config_path = Path(config_path)
    os.makedirs(config_path.parent, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path object pointing to project root
    """
    return Path(__file__).parent.parent


class Timer:
    """
    Simple context manager for timing code blocks.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, message: str = "Operation"):
        self.logger = logger
        self.message = message
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        if self.logger:
            self.logger.info(f"{self.message} completed in {elapsed:.2f} seconds")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


def format_number(num: float, decimals: int = 4) -> str:
    """
    Format a number with specified decimal places.

    Args:
        num: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{num:.{decimals}f}"


def create_experiment_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create a timestamped experiment directory.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment

    Returns:
        Path to the created experiment directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    ensure_dir(exp_dir)

    return exp_dir
