"""
Feature flags for gradual rollout and instant rollback.

This module provides runtime feature flag checks for switching between
old (environment-based) and new (injection-based) implementations.

Design Principles:
- Instant Rollback: Flag changes take effect immediately without deployment
- Fail-Safe: Defaults to new system (True), falls back gracefully
- Observability: Logs flag state on every check for debugging

Rollback Procedure:
    1. Set environment variable: USE_NEW_CONFIG=false
    2. Restart services: systemctl restart backend
    3. Expected rollback time: < 10 seconds

Usage:
    >>> from backend.app.config.feature_flags import FeatureFlags
    >>> if FeatureFlags.use_new_config_system():
    ...     service = DCOPEventDicomServiceV2(config=config)
    ... else:
    ...     service = DCOPEventDicomService()  # Legacy
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    Feature flag management for configuration system migration.

    All flags default to True (new system enabled) for forward progress.
    Set environment variables to False for instant rollback.

    Environment Variables:
        USE_NEW_CONFIG: Master switch for new config system
            - "true" (default): Use dependency injection pattern
            - "false": Fall back to os.getenv() in methods

        USE_NEW_SYNC_SERVICE: Switch for SyncService implementation
            - "true" (default): Use DCOPEventDicomServiceV2
            - "false": Use legacy DCOPEventDicomService
    """

    # Cache for flag values to reduce environment reads
    _cache: dict = {}

    @classmethod
    def _get_bool_env(
        cls, key: str, default: bool = True, use_cache: bool = True
    ) -> bool:
        """
        Get boolean value from environment variable.

        Args:
            key: Environment variable name
            default: Default value if not set
            use_cache: Whether to use cached value

        Returns:
            Boolean value from environment or default
        """
        if use_cache and key in cls._cache:
            return cls._cache[key]

        value = os.getenv(key, "").lower()

        if value in ("true", "1", "yes", "on"):
            result = True
        elif value in ("false", "0", "no", "off"):
            result = False
        else:
            result = default

        if use_cache:
            cls._cache[key] = result

        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the feature flag cache. Useful for testing."""
        cls._cache.clear()

    @classmethod
    def use_new_config_system(cls) -> bool:
        """
        Check if new configuration system should be used.

        This is the master switch for the pure function refactoring.
        When True, services receive configuration via constructor injection.
        When False, services read from environment variables directly.

        Returns:
            True if new config system should be used, False for legacy

        Environment:
            USE_NEW_CONFIG=false to disable
        """
        result = cls._get_bool_env("USE_NEW_CONFIG", default=True)
        logger.debug(f"Feature flag USE_NEW_CONFIG = {result}")
        return result

    @classmethod
    def use_new_sync_service(cls) -> bool:
        """
        Check if new SyncService (V2) should be used.

        This flag controls the SyncService implementation selection.
        Requires use_new_config_system() to also be True.

        Returns:
            True if DCOPEventDicomServiceV2 should be used

        Environment:
            USE_NEW_SYNC_SERVICE=false to disable
        """
        # New sync service requires new config system
        if not cls.use_new_config_system():
            return False

        result = cls._get_bool_env("USE_NEW_SYNC_SERVICE", default=True)
        logger.debug(f"Feature flag USE_NEW_SYNC_SERVICE = {result}")
        return result

    @classmethod
    def log_all_flags(cls) -> None:
        """
        Log current state of all feature flags.

        Call this on application startup for observability.
        """
        flags = {
            "USE_NEW_CONFIG": cls.use_new_config_system(),
            "USE_NEW_SYNC_SERVICE": cls.use_new_sync_service(),
        }
        logger.info(f"Feature flags: {flags}")


def log_feature_flags_on_startup() -> None:
    """
    Convenience function to log all feature flags.

    Call this from main.py or application startup.
    """
    FeatureFlags.log_all_flags()
