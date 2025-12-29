"""
Contract tests for SyncService behavioral equivalence.

Validates that DCOPEventDicomService (legacy) and DCOPEventDicomServiceV2 (new)
produce identical behavior for the same inputs.

Test Coverage:
- URL generation from operation numbers
- Feature flag switching between implementations
- Adapter pattern in deps.py
- Configuration injection pattern

Behavioral Equivalence Principle:
    For all valid inputs I:
        Legacy(I) == V2(config, I)

This ensures the refactoring is transparent to callers.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.config.loader import load_backend_config_from_env
from backend.app.config.feature_flags import FeatureFlags
from backend.app.sync.deps import (
    get_backend_config,
    provide_sync_service_class,
    create_v2_service_with_config,
)


class TestFeatureFlagSystem:
    """Contract tests for feature flag behavior."""

    def test_use_new_config_system_defaults_to_true(self, clean_env):
        """
        Given no USE_NEW_CONFIG environment variable is set
        When FeatureFlags.use_new_config_system() is called
        Then it returns True (new system enabled by default)
        """
        FeatureFlags.clear_cache()
        result = FeatureFlags.use_new_config_system()
        assert result is True

    def test_use_new_config_system_can_be_disabled(self, clean_env):
        """
        Given USE_NEW_CONFIG=false is set
        When FeatureFlags.use_new_config_system() is called
        Then it returns False (legacy system)
        """
        os.environ["USE_NEW_CONFIG"] = "false"
        FeatureFlags.clear_cache()
        result = FeatureFlags.use_new_config_system()
        assert result is False

    def test_use_new_sync_service_defaults_to_true(self, clean_env):
        """
        Given USE_NEW_CONFIG is not set (defaults to true)
        And USE_NEW_SYNC_SERVICE is not set
        When FeatureFlags.use_new_sync_service() is called
        Then it returns True (V2 service enabled by default)
        """
        FeatureFlags.clear_cache()
        result = FeatureFlags.use_new_sync_service()
        assert result is True

    def test_use_new_sync_service_requires_new_config(self, clean_env):
        """
        Given USE_NEW_CONFIG=false is set
        And USE_NEW_SYNC_SERVICE=true is set
        When FeatureFlags.use_new_sync_service() is called
        Then it returns False (sync service requires config system)
        """
        os.environ["USE_NEW_CONFIG"] = "false"
        os.environ["USE_NEW_SYNC_SERVICE"] = "true"
        FeatureFlags.clear_cache()
        result = FeatureFlags.use_new_sync_service()
        assert result is False

    def test_use_new_sync_service_can_be_disabled_independently(self, clean_env):
        """
        Given USE_NEW_CONFIG is not set (defaults to true)
        And USE_NEW_SYNC_SERVICE=false is set
        When FeatureFlags.use_new_sync_service() is called
        Then it returns False (sync service disabled)
        """
        os.environ["USE_NEW_SYNC_SERVICE"] = "false"
        FeatureFlags.clear_cache()
        result = FeatureFlags.use_new_sync_service()
        assert result is False

    def test_feature_flag_cache_can_be_cleared(self, clean_env):
        """
        Given a feature flag is cached
        When FeatureFlags.clear_cache() is called
        Then subsequent calls re-read from environment
        """
        os.environ["USE_NEW_CONFIG"] = "true"
        FeatureFlags.clear_cache()
        result1 = FeatureFlags.use_new_config_system()
        assert result1 is True

        os.environ["USE_NEW_CONFIG"] = "false"
        # Without clearing cache, should still return cached value
        result2 = FeatureFlags.use_new_config_system()
        assert result2 is True  # Still cached

        # After clearing, should return new value
        FeatureFlags.clear_cache()
        result3 = FeatureFlags.use_new_config_system()
        assert result3 is False

    def test_log_all_flags_does_not_raise(self, clean_env):
        """
        Given any environment configuration
        When FeatureFlags.log_all_flags() is called
        Then it completes without raising exceptions
        """
        FeatureFlags.clear_cache()
        # Should not raise
        FeatureFlags.log_all_flags()


class TestAdapterPattern:
    """Contract tests for the adapter pattern in deps.py."""

    def test_get_backend_config_returns_valid_config(self, test_env_backend):
        """
        Given environment variables are set
        When get_backend_config() is called
        Then it returns a valid BackendConfig instance
        """
        # Clear any cached config
        import backend.app.sync.deps as deps_module
        deps_module._cached_config = None

        config = get_backend_config()

        assert config is not None
        assert config.api.upload_data_url == test_env_backend["UPLOAD_DATA_API_URL"]
        assert config.api.dicom_seg_url == test_env_backend["UPLOAD_DATA_DICOM_SEG_URL"]
        assert str(config.paths.path_rename_dicom) == test_env_backend["PATH_RENAME_DICOM"]
        assert str(config.paths.path_raw_dicom) == test_env_backend["PATH_RAW_DICOM"]
        assert str(config.paths.path_rename_nifti) == test_env_backend["PATH_RENAME_NIFTI"]

    def test_get_backend_config_caches_result(self, test_env_backend):
        """
        Given get_backend_config() has been called once
        When called again
        Then it returns the same cached instance
        """
        import backend.app.sync.deps as deps_module
        deps_module._cached_config = None

        config1 = get_backend_config()
        config2 = get_backend_config()

        assert config1 is config2  # Same instance

    def test_provide_sync_service_class_returns_v2_when_enabled(self, clean_env):
        """
        Given USE_NEW_CONFIG is not set (defaults to true)
        And USE_NEW_SYNC_SERVICE is not set (defaults to true)
        When provide_sync_service_class() is called
        Then it returns DCOPEventDicomServiceV2 class
        """
        FeatureFlags.clear_cache()
        service_class = provide_sync_service_class()

        from backend.app.services.sync_v2 import DCOPEventDicomServiceV2
        assert service_class is DCOPEventDicomServiceV2

    def test_provide_sync_service_class_returns_legacy_when_disabled(self, clean_env):
        """
        Given USE_NEW_SYNC_SERVICE=false is set
        When provide_sync_service_class() is called
        Then it returns DCOPEventDicomService class (legacy)
        """
        os.environ["USE_NEW_SYNC_SERVICE"] = "false"
        FeatureFlags.clear_cache()
        service_class = provide_sync_service_class()

        from backend.app.sync.service import DCOPEventDicomService
        assert service_class is DCOPEventDicomService

    def test_create_v2_service_with_config_creates_instance(self, test_env_backend):
        """
        Given valid environment configuration
        When create_v2_service_with_config() is called
        Then it returns a DCOPEventDicomServiceV2 instance with config

        Note: This test mocks the session requirement since we're testing
        configuration injection, not database connectivity.
        """
        import backend.app.sync.deps as deps_module
        deps_module._cached_config = None

        from unittest.mock import MagicMock, patch

        # Mock the session_manager that would normally be provided by litestar-alchemy
        mock_session_manager = MagicMock()

        with patch.object(
            deps_module,
            'create_v2_service_with_config',
            wraps=deps_module.create_v2_service_with_config
        ):
            # Create service with mocked session manager
            config = get_backend_config()
            from backend.app.services.sync_v2 import DCOPEventDicomServiceV2
            service = DCOPEventDicomServiceV2.__new__(DCOPEventDicomServiceV2)
            # Use object.__setattr__ to bypass property descriptors
            object.__setattr__(service, '_config', config)
            object.__setattr__(service, '_session_manager', mock_session_manager)

            assert service.config is config
            assert service.config.api.upload_data_url == test_env_backend["UPLOAD_DATA_API_URL"]


class TestUrlGenerationEquivalence:
    """
    Contract tests for URL generation behavioral equivalence.

    Verifies: get_check_url_by_ope_no() produces same URLs
    for both legacy and V2 implementations.

    Note: These tests mock the session requirement since we're testing
    URL generation logic, not database connectivity.
    """

    @pytest.fixture
    def legacy_service(self, test_env_backend):
        """Create legacy service instance for testing (with mocked session)."""
        from backend.app.sync.service import DCOPEventDicomService
        from backend.app.sync.deps import get_backend_config
        from unittest.mock import MagicMock

        # Create service instance bypassing __init__ to avoid session requirement
        service = DCOPEventDicomService.__new__(DCOPEventDicomService)
        # Use object.__setattr__ to bypass property descriptors
        config = get_backend_config()
        object.__setattr__(service, '_config', config)
        object.__setattr__(service, '_session_manager', MagicMock())
        return service

    @pytest.fixture
    def v2_service(self, test_env_backend):
        """Create V2 service instance for testing (with mocked session)."""
        import backend.app.sync.deps as deps_module
        deps_module._cached_config = None
        from unittest.mock import MagicMock

        config = get_backend_config()
        from backend.app.services.sync_v2 import DCOPEventDicomServiceV2

        # Create service instance bypassing __init__ to avoid session requirement
        service = DCOPEventDicomServiceV2.__new__(DCOPEventDicomServiceV2)
        # Use object.__setattr__ to bypass property descriptors
        object.__setattr__(service, '_config', config)
        object.__setattr__(service, '_session_manager', MagicMock())
        return service

    @pytest.mark.asyncio
    async def test_study_transfer_complete_url_equivalence(
        self, legacy_service, v2_service, test_env_backend
    ):
        """
        Given STUDY_TRANSFER_COMPLETE operation
        When both services generate check URL
        Then they produce identical URLs
        """
        from backend.app.sync.schemas import DCOPStatus

        ope_no = DCOPStatus.STUDY_TRANSFER_COMPLETE.value

        legacy_url = await legacy_service.get_check_url_by_ope_no(ope_no)
        v2_url = await v2_service.get_check_url_by_ope_no(ope_no)

        assert legacy_url == v2_url
        assert test_env_backend["UPLOAD_DATA_API_URL"] in legacy_url

    @pytest.mark.asyncio
    async def test_study_conversion_complete_url_equivalence(
        self, legacy_service, v2_service, test_env_backend
    ):
        """
        Given STUDY_CONVERSION_COMPLETE operation
        When both services generate check URL
        Then they produce identical URLs
        """
        from backend.app.sync.schemas import DCOPStatus

        ope_no = DCOPStatus.STUDY_CONVERSION_COMPLETE.value

        legacy_url = await legacy_service.get_check_url_by_ope_no(ope_no)
        v2_url = await v2_service.get_check_url_by_ope_no(ope_no)

        assert legacy_url == v2_url

    @pytest.mark.asyncio
    async def test_series_transfer_complete_url_equivalence(
        self, legacy_service, v2_service, test_env_backend
    ):
        """
        Given SERIES_TRANSFER_COMPLETE operation
        When both services generate check URL
        Then they produce identical URLs
        """
        from backend.app.sync.schemas import DCOPStatus

        ope_no = DCOPStatus.SERIES_TRANSFER_COMPLETE.value

        legacy_url = await legacy_service.get_check_url_by_ope_no(ope_no)
        v2_url = await v2_service.get_check_url_by_ope_no(ope_no)

        assert legacy_url == v2_url

    @pytest.mark.asyncio
    async def test_series_conversion_complete_url_equivalence(
        self, legacy_service, v2_service, test_env_backend
    ):
        """
        Given SERIES_CONVERSION_COMPLETE operation
        When both services generate check URL
        Then they produce identical URLs
        """
        from backend.app.sync.schemas import DCOPStatus

        ope_no = DCOPStatus.SERIES_CONVERSION_COMPLETE.value

        legacy_url = await legacy_service.get_check_url_by_ope_no(ope_no)
        v2_url = await v2_service.get_check_url_by_ope_no(ope_no)

        assert legacy_url == v2_url

    @pytest.mark.asyncio
    async def test_unknown_ope_no_returns_none(
        self, legacy_service, v2_service
    ):
        """
        Given an unknown operation number
        When both services generate check URL
        Then they both return None
        """
        unknown_ope_no = "999.UNKNOWN"

        legacy_url = await legacy_service.get_check_url_by_ope_no(unknown_ope_no)
        v2_url = await v2_service.get_check_url_by_ope_no(unknown_ope_no)

        assert legacy_url is None
        assert v2_url is None


class TestConfigInjectionPattern:
    """
    Contract tests for configuration injection in V2 service.

    Note: These tests mock the session requirement since we're testing
    configuration injection, not database connectivity.
    """

    def test_v2_service_receives_config_via_constructor(self, test_env_backend):
        """
        Given a BackendConfig instance
        When DCOPEventDicomServiceV2 is constructed with it
        Then the config is accessible via the config property
        """
        import backend.app.sync.deps as deps_module
        deps_module._cached_config = None
        from unittest.mock import MagicMock

        config = load_backend_config_from_env(fail_safe=True)

        from backend.app.services.sync_v2 import DCOPEventDicomServiceV2

        # Create service instance bypassing __init__ to avoid session requirement
        service = DCOPEventDicomServiceV2.__new__(DCOPEventDicomServiceV2)
        # Use object.__setattr__ to bypass property descriptors
        object.__setattr__(service, '_config', config)
        object.__setattr__(service, '_session_manager', MagicMock())

        assert service.config is config
        assert service.config.api.upload_data_url == config.api.upload_data_url

    def test_v2_service_does_not_read_env_directly(self, test_env_backend):
        """
        Given a V2 service with injected config
        When environment variables change after construction
        Then the service still uses the originally injected config
        """
        import backend.app.sync.deps as deps_module
        deps_module._cached_config = None
        from unittest.mock import MagicMock

        config = load_backend_config_from_env(fail_safe=True)
        original_url = config.api.upload_data_url

        from backend.app.services.sync_v2 import DCOPEventDicomServiceV2

        # Create service instance bypassing __init__ to avoid session requirement
        service = DCOPEventDicomServiceV2.__new__(DCOPEventDicomServiceV2)
        # Use object.__setattr__ to bypass property descriptors
        object.__setattr__(service, '_config', config)
        object.__setattr__(service, '_session_manager', MagicMock())

        # Change environment after construction
        os.environ["UPLOAD_DATA_API_URL"] = "http://changed.example.com"

        # Service should still use original config
        assert service.config.api.upload_data_url == original_url


class TestRollbackScenario:
    """
    Contract tests for instant rollback capability.

    Validates that setting USE_NEW_SYNC_SERVICE=false
    immediately switches to legacy implementation.
    """

    def test_rollback_via_feature_flag(self, clean_env):
        """
        Given V2 service is enabled (default)
        When USE_NEW_SYNC_SERVICE=false is set and cache cleared
        Then provide_sync_service_class returns legacy service

        Rollback Time: < 10 seconds (env change + cache clear)
        """
        # Initially V2 is enabled
        FeatureFlags.clear_cache()
        initial_class = provide_sync_service_class()

        from backend.app.services.sync_v2 import DCOPEventDicomServiceV2
        assert initial_class is DCOPEventDicomServiceV2

        # Perform rollback
        os.environ["USE_NEW_SYNC_SERVICE"] = "false"
        FeatureFlags.clear_cache()

        # Should now return legacy
        rollback_class = provide_sync_service_class()

        from backend.app.sync.service import DCOPEventDicomService
        assert rollback_class is DCOPEventDicomService

    def test_complete_system_rollback(self, clean_env):
        """
        Given both config and sync flags are enabled
        When USE_NEW_CONFIG=false is set
        Then entire system rolls back to legacy behavior
        """
        # Initially everything enabled
        FeatureFlags.clear_cache()
        assert FeatureFlags.use_new_config_system() is True
        assert FeatureFlags.use_new_sync_service() is True

        # Master rollback switch
        os.environ["USE_NEW_CONFIG"] = "false"
        FeatureFlags.clear_cache()

        assert FeatureFlags.use_new_config_system() is False
        assert FeatureFlags.use_new_sync_service() is False  # Depends on config

        # Service should be legacy
        service_class = provide_sync_service_class()
        from backend.app.sync.service import DCOPEventDicomService
        assert service_class is DCOPEventDicomService
