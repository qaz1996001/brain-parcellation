# app/routers/sync/deps.py
"""
Dependency injection for sync services.

This module provides service factory functions that use feature flags
to select between legacy and V2 implementations.

Design Pattern: Adapter Pattern via Dependency Injection
- Feature flag check at dependency resolution time
- Transparent service switching without router changes
- Backward compatibility: old routes continue to work unchanged

Usage in routers:
    @router.post("/example")
    async def example(
        service: Annotated[DCOPEventDicomService, Depends(provide_sync_service)],
    ):
        # Works with both legacy and V2 implementations
        await service.some_method()
"""

import logging
from typing import Any, Generator, Optional, Union

from backend.app.config.feature_flags import FeatureFlags
from backend.app.config.loader import load_backend_config_from_env
from backend.app.config.models import BackendConfig

logger = logging.getLogger(__name__)

# Cached configuration instance (loaded once at startup)
_cached_config: Optional[BackendConfig] = None


def get_backend_config() -> BackendConfig:
    """
    Get cached backend configuration.

    Uses fail-safe mode for production reliability.
    Configuration is loaded once and cached for the application lifetime.

    Returns:
        BackendConfig: Immutable configuration instance
    """
    global _cached_config
    if _cached_config is None:
        _cached_config = load_backend_config_from_env(fail_safe=True)
        logger.info("Backend configuration loaded and cached")
    return _cached_config


def provide_sync_service_class() -> type:
    """
    Provide the appropriate sync service class based on feature flags.

    This function is used with alchemy.provide_service() to dynamically
    select between legacy and V2 implementations.

    Returns:
        Service class (DCOPEventDicomService or DCOPEventDicomServiceV2)

    Note:
        Currently returns legacy service to maintain exact backward compatibility.
        V2 service can be enabled via USE_NEW_SYNC_SERVICE=true after testing.
    """
    if FeatureFlags.use_new_sync_service():
        from backend.app.services.sync_v2 import DCOPEventDicomServiceV2

        logger.debug("Using DCOPEventDicomServiceV2 (new implementation)")
        return DCOPEventDicomServiceV2
    else:
        from backend.app.sync.service import DCOPEventDicomService

        logger.debug("Using DCOPEventDicomService (legacy implementation)")
        return DCOPEventDicomService


def create_v2_service_with_config(**kwargs: Any):
    """
    Factory function to create V2 service with injected configuration.

    This is used when explicitly creating V2 service instances outside
    of the alchemy dependency injection framework.

    Args:
        **kwargs: Additional arguments passed to service constructor

    Returns:
        DCOPEventDicomServiceV2: Configured V2 service instance
    """
    from backend.app.services.sync_v2 import DCOPEventDicomServiceV2

    config = get_backend_config()
    return DCOPEventDicomServiceV2(config=config, **kwargs)
