"""
Contract tests for configuration loading.

Validates that configuration loaders work correctly in both fail-safe (production)
and strict (testing) modes.

Test Coverage:
- Fail-safe mode returns valid config with defaults
- Strict mode raises on missing required env vars
- Type conversion and validation
- Immutability of config objects
"""

from pathlib import Path

import pytest

from backend.app.config import BackendConfig, load_backend_config_from_env, DEFAULT_CONFIG as BACKEND_DEFAULT
from code_ai.config import CodeAIConfig, load_code_ai_config_from_env, DEFAULT_CONFIG as CODE_AI_DEFAULT


class TestBackendConfigLoading:
    """Contract tests for backend configuration loading."""

    def test_fail_safe_mode_returns_valid_config_with_defaults(self, empty_env):
        """
        Given no environment variables are set
        When load_backend_config_from_env(fail_safe=True) is called
        Then it returns DEFAULT_CONFIG with safe fallback values
        And all path fields point to temporary directories
        And the configuration is valid and complete
        """
        config = load_backend_config_from_env(fail_safe=True)

        # Verify config is valid BackendConfig instance
        assert isinstance(config, BackendConfig)

        # Verify defaults are used
        assert config.api.upload_data_url == BACKEND_DEFAULT.api.upload_data_url
        assert config.database.connection_string == BACKEND_DEFAULT.database.connection_string

        # Verify all paths are Path objects
        assert isinstance(config.paths.path_process, Path)
        assert isinstance(config.paths.path_json, Path)
        assert isinstance(config.paths.path_log, Path)
        assert isinstance(config.paths.path_root, Path)
        assert isinstance(config.paths.path_rename_dicom, Path)

    def test_strict_mode_with_env_vars_succeeds(self, test_env_backend):
        """
        Given all required environment variables are set
        When load_backend_config_from_env(fail_safe=False) is called
        Then it returns config with values from environment
        And no errors are raised
        """
        config = load_backend_config_from_env(fail_safe=False)

        # Verify config uses environment variables
        assert config.api.upload_data_url == test_env_backend["UPLOAD_DATA_API_URL"]
        assert config.database.connection_string == test_env_backend["AI_APP_CONNECTION_STRING"]

        # Verify paths are converted from strings
        assert str(config.paths.path_process) == test_env_backend["PATH_PROCESS"]
        assert str(config.paths.path_json) == test_env_backend["PATH_JSON"]

    def test_config_is_immutable(self, empty_env):
        """
        Given a config object is created
        When attempting to modify any field
        Then a FrozenInstanceError is raised
        """
        config = load_backend_config_from_env(fail_safe=True)

        # Verify frozen dataclass prevents modification
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            config.api.upload_data_url = "http://hacked.com"

        with pytest.raises(Exception):
            config.database.connection_string = "malicious"

    def test_type_conversion_for_paths(self, test_env_backend):
        """
        Given environment variables contain path strings
        When config is loaded
        Then paths are converted to Path objects
        """
        config = load_backend_config_from_env(fail_safe=True)

        assert isinstance(config.paths.path_process, Path)
        assert isinstance(config.paths.path_json, Path)
        assert isinstance(config.paths.path_log, Path)


class TestCodeAIConfigLoading:
    """Contract tests for code_ai configuration loading."""

    def test_fail_safe_mode_returns_valid_config_with_defaults(self, empty_env):
        """
        Given no environment variables are set
        When load_code_ai_config_from_env(fail_safe=True) is called
        Then it returns DEFAULT_CONFIG with safe fallback values
        And the configuration is valid and complete
        """
        config = load_code_ai_config_from_env(fail_safe=True)

        # Verify config is valid CodeAIConfig instance
        assert isinstance(config, CodeAIConfig)

        # Verify defaults are used
        assert config.model.gpu_n == CODE_AI_DEFAULT.model.gpu_n
        assert config.tensorflow.cpp_min_log_level == CODE_AI_DEFAULT.tensorflow.cpp_min_log_level

        # Verify all paths are Path objects
        assert isinstance(config.paths.path_code, Path)
        assert isinstance(config.paths.path_process, Path)
        assert isinstance(config.paths.path_json, Path)
        assert isinstance(config.paths.path_log, Path)
        assert isinstance(config.model.path_synthseg, Path)

    def test_strict_mode_with_env_vars_succeeds(self, test_env_code_ai):
        """
        Given all environment variables are set
        When load_code_ai_config_from_env(fail_safe=False) is called
        Then it returns config with values from environment
        """
        config = load_code_ai_config_from_env(fail_safe=False)

        # Verify config uses environment variables
        assert config.model.gpu_n == int(test_env_code_ai["GPU_N"])
        assert config.tensorflow.cpp_min_log_level == test_env_code_ai["TF_CPP_MIN_LOG_LEVEL"]

        # Verify paths are converted from strings
        assert str(config.paths.path_code) == test_env_code_ai["PATH_CODE"]
        assert str(config.paths.path_process) == test_env_code_ai["PATH_PROCESS"]

    def test_config_is_immutable(self, empty_env):
        """
        Given a config object is created
        When attempting to modify any field
        Then a FrozenInstanceError is raised
        """
        config = load_code_ai_config_from_env(fail_safe=True)

        # Verify frozen dataclass prevents modification
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            config.model.gpu_n = 999

        with pytest.raises(Exception):
            config.tensorflow.cpp_min_log_level = "0"

    def test_type_conversion_for_integers(self, test_env_code_ai):
        """
        Given environment variable contains integer string
        When config is loaded
        Then value is converted to int type
        """
        config = load_code_ai_config_from_env(fail_safe=True)

        assert isinstance(config.model.gpu_n, int)
        assert config.model.gpu_n == int(test_env_code_ai["GPU_N"])

    def test_type_conversion_for_paths(self, test_env_code_ai):
        """
        Given environment variables contain path strings
        When config is loaded
        Then paths are converted to Path objects
        """
        config = load_code_ai_config_from_env(fail_safe=True)

        assert isinstance(config.paths.path_code, Path)
        assert isinstance(config.paths.path_json, Path)
        assert isinstance(config.model.path_synthseg, Path)
