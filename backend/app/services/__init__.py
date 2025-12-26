"""
Services module with dependency injection pattern.

This module contains V2 service implementations that receive configuration
via constructor injection instead of reading from environment variables.

Design Pattern: Adapter Pattern for Backward Compatibility
- V2 services: New implementations using dependency injection
- Adapters: Preserve old interface, delegate to V2 internally
- Feature flags: Runtime switch between implementations
"""

from .sync_v2 import DCOPEventDicomServiceV2

__all__ = ["DCOPEventDicomServiceV2"]
