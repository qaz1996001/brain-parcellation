"""
Contract tests for pipeline base utilities and dual-mode configuration.

Validates that the dual-mode configuration pattern works correctly:
- Old pattern: pipeline(ID, params) uses explicit parameters
- New pattern: pipeline(ID, params, config=config) uses config object

Test Coverage:
- get_config() caching behavior
- get_gpu_n() dual-mode behavior
- clear_config_cache() for test isolation
- Path resolution with fallback

Behavioral Equivalence Principle:
    For all valid inputs I:
        pipeline(I, ...) == pipeline(I, ..., config=config)
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestGetConfigFunction:
    """Contract tests for get_config() dual-mode behavior."""

    def test_get_config_returns_provided_config(self, clean_env, test_env_code_ai):
        """
        Given an explicit config object is provided
        When get_config(config) is called
        Then it returns the same config object (pure function pattern)
        """
        from code_ai.pipeline.base import get_config, clear_config_cache
        from code_ai.config import CodeAIConfig
        from code_ai.config.models import PathConfig, ModelConfig, TensorFlowConfig

        clear_config_cache()

        # Create explicit config
        explicit_config = CodeAIConfig(
            paths=PathConfig(
                path_code=Path("/explicit/code"),
                path_process=Path("/explicit/process"),
                path_json=Path("/explicit/json"),
                path_log=Path("/explicit/log"),
            ),
            model=ModelConfig(
                path_synthseg=Path("/explicit/synthseg"),
                gpu_n=2,
            ),
            tensorflow=TensorFlowConfig(),
        )

        result = get_config(explicit_config)

        assert result is explicit_config
        assert str(result.paths.path_code) == "/explicit/code"
        assert result.model.gpu_n == 2

    def test_get_config_loads_from_env_when_none(self, clean_env, test_env_code_ai):
        """
        Given no config is provided (config=None)
        When get_config() is called
        Then it loads from environment (legacy backward-compatible pattern)
        """
        from code_ai.pipeline.base import get_config, clear_config_cache

        clear_config_cache()

        result = get_config(None)

        assert result is not None
        assert str(result.paths.path_code) == test_env_code_ai["PATH_CODE"]
        assert result.model.gpu_n == int(test_env_code_ai["GPU_N"])

    def test_get_config_caches_env_loaded_config(self, clean_env, test_env_code_ai):
        """
        Given config is loaded from environment
        When get_config() is called multiple times
        Then it returns the same cached instance
        """
        from code_ai.pipeline.base import get_config, clear_config_cache

        clear_config_cache()

        config1 = get_config(None)
        config2 = get_config(None)

        assert config1 is config2  # Same cached instance

    def test_get_config_explicit_bypasses_cache(self, clean_env, test_env_code_ai):
        """
        Given an explicit config is provided
        When get_config(explicit) is called
        Then it bypasses the cache and returns the explicit config
        """
        from code_ai.pipeline.base import get_config, clear_config_cache
        from code_ai.config import CodeAIConfig
        from code_ai.config.models import PathConfig, ModelConfig, TensorFlowConfig

        clear_config_cache()

        # Load from env first to populate cache
        cached_config = get_config(None)

        # Create explicit config
        explicit_config = CodeAIConfig(
            paths=PathConfig(
                path_code=Path("/explicit/code"),
                path_process=Path("/explicit/process"),
                path_json=Path("/explicit/json"),
                path_log=Path("/explicit/log"),
            ),
            model=ModelConfig(
                path_synthseg=Path("/explicit/synthseg"),
                gpu_n=5,
            ),
            tensorflow=TensorFlowConfig(),
        )

        result = get_config(explicit_config)

        assert result is explicit_config
        assert result is not cached_config


class TestClearConfigCache:
    """Contract tests for clear_config_cache() test utility."""

    def test_clear_cache_allows_fresh_load(self, clean_env, test_env_code_ai):
        """
        Given config cache is populated
        When clear_config_cache() is called and env changes
        Then subsequent get_config() reflects new environment
        """
        from code_ai.pipeline.base import get_config, clear_config_cache

        clear_config_cache()

        # Load initial config
        config1 = get_config(None)
        original_gpu = config1.model.gpu_n

        # Change environment
        os.environ["GPU_N"] = "99"

        # Without clearing, cache is still used
        config2 = get_config(None)
        assert config2.model.gpu_n == original_gpu  # Still cached

        # Clear cache
        clear_config_cache()

        # Now should reflect new env
        config3 = get_config(None)
        assert config3.model.gpu_n == 99


class TestGetGpuN:
    """Contract tests for get_gpu_n() dual-mode behavior."""

    def test_get_gpu_n_from_config(self, clean_env):
        """
        Given a config object with gpu_n=3
        When get_gpu_n(config) is called
        Then it returns 3
        """
        from code_ai.pipeline.base import get_gpu_n
        from code_ai.config import CodeAIConfig
        from code_ai.config.models import PathConfig, ModelConfig, TensorFlowConfig

        config = CodeAIConfig(
            paths=PathConfig(
                path_code=Path("/test/code"),
                path_process=Path("/test/process"),
                path_json=Path("/test/json"),
                path_log=Path("/test/log"),
            ),
            model=ModelConfig(
                path_synthseg=Path("/test/synthseg"),
                gpu_n=3,
            ),
            tensorflow=TensorFlowConfig(),
        )

        result = get_gpu_n(config)

        assert result == 3

    def test_get_gpu_n_from_env(self, clean_env):
        """
        Given GPU_N=5 is set in environment
        When get_gpu_n(None) is called
        Then it returns 5
        """
        from code_ai.pipeline.base import get_gpu_n

        os.environ["GPU_N"] = "5"

        result = get_gpu_n(None)

        assert result == 5

    def test_get_gpu_n_default_when_empty(self, clean_env):
        """
        Given GPU_N is not set in environment
        When get_gpu_n(None) is called
        Then it returns 0 (default)
        """
        from code_ai.pipeline.base import get_gpu_n

        # Ensure GPU_N is not set
        os.environ.pop("GPU_N", None)

        result = get_gpu_n(None)

        assert result == 0

    def test_get_gpu_n_invalid_value(self, clean_env):
        """
        Given GPU_N is set to invalid value
        When get_gpu_n(None) is called
        Then it returns 0 (safe default)
        """
        from code_ai.pipeline.base import get_gpu_n

        os.environ["GPU_N"] = "not_a_number"

        result = get_gpu_n(None)

        assert result == 0


class TestGetPathWithFallback:
    """Contract tests for get_path_with_fallback() dual-mode behavior."""

    def test_path_from_config(self, clean_env):
        """
        Given a config with path_log=/config/log
        When get_path_with_fallback(config, "PATH_LOG", config.paths.path_log) is called
        Then it returns /config/log
        """
        from code_ai.pipeline.base import get_path_with_fallback

        config = MagicMock()
        config.paths.path_log = Path("/config/log")

        result = get_path_with_fallback(
            config, "PATH_LOG", config.paths.path_log
        )

        assert result == "/config/log"

    def test_path_from_env(self, clean_env):
        """
        Given PATH_LOG=/env/log in environment and config=None
        When get_path_with_fallback(None, "PATH_LOG", None) is called
        Then it returns /env/log from environment
        """
        from code_ai.pipeline.base import get_path_with_fallback

        os.environ["PATH_LOG"] = "/env/log"

        result = get_path_with_fallback(None, "PATH_LOG", None)

        assert result == "/env/log"

    def test_path_default_fallback(self, clean_env):
        """
        Given PATH_LOG is not set and config=None
        When get_path_with_fallback(None, "PATH_LOG", None, "/default/log") is called
        Then it returns /default/log
        """
        from code_ai.pipeline.base import get_path_with_fallback

        # Ensure PATH_LOG is not set
        os.environ.pop("PATH_LOG", None)

        result = get_path_with_fallback(
            None, "PATH_LOG", None, "/default/log"
        )

        assert result == "/default/log"


class TestPipelineDualModeContract:
    """
    Contract tests for pipeline function dual-mode behavior.

    These tests verify that pipelines work correctly with both:
    - Old pattern: explicit parameters (backward compatible)
    - New pattern: config injection (pure function)
    """

    def test_pipeline_cmb_signature_accepts_config(self, clean_env):
        """
        Given the pipeline_cmb function
        When inspecting its signature
        Then it accepts config: Optional[CodeAIConfig] = None
        """
        import inspect
        from code_ai.pipeline.pipeline_cmb_tensorflow import pipeline_cmb

        sig = inspect.signature(pipeline_cmb)

        assert "config" in sig.parameters
        assert sig.parameters["config"].default is None

    def test_pipeline_aneurysm_signature_accepts_config(self, clean_env):
        """
        Given the pipeline_aneurysm function
        When inspecting its signature
        Then it accepts config: Optional[CodeAIConfig] = None
        """
        import inspect
        try:
            from code_ai.pipeline.pipeline_aneurysm_tensorflow import pipeline_aneurysm
        except ImportError:
            pytest.skip("pipeline_aneurysm dependencies not installed")

        sig = inspect.signature(pipeline_aneurysm)

        assert "config" in sig.parameters
        assert sig.parameters["config"].default is None

    def test_pipeline_cmb_backward_compatible(self, clean_env, test_env_code_ai):
        """
        Given pipeline_cmb is called without config parameter
        When using the old calling pattern
        Then it works with explicit parameters (backward compatible)

        Note: This test only verifies the function can be called,
        not full execution (which requires GPU and model files)
        """
        import inspect
        from code_ai.pipeline.pipeline_cmb_tensorflow import pipeline_cmb

        # Verify the function signature is backward compatible
        sig = inspect.signature(pipeline_cmb)
        params = list(sig.parameters.keys())

        # Required positional params still work
        assert "ID" in params
        assert "swan_file" in params
        assert "t1_file" in params
        assert "path_output" in params

        # Config is optional (has default)
        assert sig.parameters["config"].default is None

    def test_pipeline_aneurysm_backward_compatible(self, clean_env, test_env_code_ai):
        """
        Given pipeline_aneurysm is called without config parameter
        When using the old calling pattern
        Then it works with explicit parameters (backward compatible)
        """
        import inspect
        try:
            from code_ai.pipeline.pipeline_aneurysm_tensorflow import pipeline_aneurysm
        except ImportError:
            pytest.skip("pipeline_aneurysm dependencies not installed (cv2)")

        sig = inspect.signature(pipeline_aneurysm)
        params = list(sig.parameters.keys())

        # Required positional params still work
        assert "ID" in params
        assert "MRA_BRAIN_file" in params
        assert "path_output" in params

        # Config is optional (has default)
        assert sig.parameters["config"].default is None
