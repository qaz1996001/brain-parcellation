"""
Environment configuration for production and testing environments.

Design principles:
- Ken Thompson: Simple, single control point via ENV variable
- Linus Torvalds: Data structure first, configuration as data
- Martin Fowler: YAGNI - only production and testing
- Donald Knuth: Precise boundaries, provable correctness
"""

import os
from typing import TypedDict, Literal


# Type definitions for environment and configuration
Environment = Literal["production", "testing"]


class EnvironmentConfig(TypedDict):
    """
    Environment-specific configuration schema.

    Attributes:
        data_root: Root directory for data storage
        log_level: Logging level for the environment
        model_config_path: Path to model configuration file
    """
    data_root: str
    log_level: str
    model_config_path: str


# Layer 2: Environment-specific configuration mapping (Linus: data structure first)
ENVIRONMENT_CONFIGS: dict[Environment, EnvironmentConfig] = {
    "production": {
        "data_root": "/data/production",
        "log_level": "INFO",
        "model_config_path": "/models/production/config.yaml",
    },
    "testing": {
        "data_root": "/data/testing",
        "log_level": "DEBUG",
        "model_config_path": "/models/testing/config.yaml",
    },
}


class EnvironmentError(Exception):
    """Raised when environment configuration is invalid."""
    pass


def get_environment() -> Environment:
    """
    Get the current environment from ENV environment variable.

    Following Twelve-Factor App principle: configuration via environment variables.
    Default to production for conservative, safe behavior (Ken Thompson: simplicity).

    Returns:
        Current environment ("production" or "testing")

    Raises:
        EnvironmentError: If ENV contains invalid value

    Examples:
        >>> os.environ["ENV"] = "testing"
        >>> get_environment()
        'testing'

        >>> os.environ.pop("ENV", None)
        >>> get_environment()  # Defaults to production
        'production'
    """
    # Layer 1: Environment variable (highest priority)
    env = os.getenv("ENV", "production")

    # Knuth: Precise validation with clear error messages
    if env not in ("production", "testing"):
        raise EnvironmentError(
            f"Invalid environment: ENV={env}. "
            f"Valid values: production, testing"
        )

    return env  # type: ignore


def get_config() -> EnvironmentConfig:
    """
    Get configuration for the current environment.

    Layer 3: Runtime configuration resolution based on environment.
    Configuration is immutable after application startup.

    Returns:
        Environment-specific configuration dictionary

    Raises:
        EnvironmentError: If environment is invalid or configuration missing

    Examples:
        >>> os.environ["ENV"] = "testing"
        >>> config = get_config()
        >>> config["log_level"]
        'DEBUG'
    """
    env = get_environment()

    # Knuth: Verify configuration exists (defensive programming)
    if env not in ENVIRONMENT_CONFIGS:
        raise EnvironmentError(
            f"Configuration missing for environment: {env}"
        )

    config = ENVIRONMENT_CONFIGS[env]

    # Knuth: Validate configuration schema completeness
    required_keys = {"data_root", "log_level", "model_config_path"}
    actual_keys = set(config.keys())

    if actual_keys != required_keys:
        missing = required_keys - actual_keys
        extra = actual_keys - required_keys
        raise EnvironmentError(
            f"Invalid configuration schema for {env}. "
            f"Missing keys: {missing}, Extra keys: {extra}"
        )

    return config


def validate_environment(expected: Environment) -> None:
    """
    Validate that the current environment matches expected value.

    Knuth-style validation: explicit, provable correctness.
    Use this for environment-specific operations that should only run
    in specific environments (e.g., data migrations, destructive operations).

    Args:
        expected: Expected environment value

    Raises:
        EnvironmentError: If current environment doesn't match expected

    Examples:
        >>> os.environ["ENV"] = "production"
        >>> validate_environment("production")  # Passes

        >>> os.environ["ENV"] = "testing"
        >>> validate_environment("production")  # Raises EnvironmentError
        Traceback (most recent call last):
        ...
        EnvironmentError: Environment mismatch: expected=production, actual=testing
    """
    actual = get_environment()

    if actual != expected:
        raise EnvironmentError(
            f"Environment mismatch: expected={expected}, actual={actual}. "
            f"This operation requires ENV={expected}"
        )
