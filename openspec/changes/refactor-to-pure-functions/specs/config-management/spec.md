# Spec: Config Management

## ADDED Requirements

### Requirement: Centralized Configuration Loading

All environment variable reads MUST be concentrated in dedicated configuration loader functions. The system MUST provide exactly two entry points for environment configuration: one for backend and one for code_ai.

#### Scenario: Backend configuration loading
```python
Given the backend application needs configuration
When the system calls load_backend_config_from_env()
Then all environment variables are read in one place
And a BackendConfig dataclass is returned
And the config object is immutable (frozen dataclass)
```

#### Scenario: Code_AI configuration loading
```python
Given the code_ai application needs configuration
When the system calls load_code_ai_config_from_env()
Then all environment variables are read in one place
And a CodeAIConfig dataclass is returned
And the config object is immutable (frozen dataclass)
```

#### Scenario: Zero scattered environment reads
```python
Given the configuration loaders are implemented
When searching the codebase for os.getenv() calls
Then only 2 instances should exist: in loader functions only
And zero instances should exist in service or pipeline code
```

### Requirement: Type-Safe Configuration Dataclasses

All configuration MUST be represented as immutable, type-annotated dataclasses with explicit field definitions.

#### Scenario: Backend configuration structure
```python
Given the backend configuration is defined
When inspecting the BackendConfig dataclass
Then it MUST be decorated with @dataclass(frozen=True)
And it MUST contain typed fields: api, paths, database
And each field MUST have a type annotation
And each nested config MUST also be a frozen dataclass
```

#### Scenario: Type validation at runtime
```python
Given a configuration dataclass definition
When the loader creates an instance with invalid types
Then the system MUST raise a TypeError
And the error MUST identify the invalid field
```

### Requirement: Environment-Aware Configuration Loading

Configuration loaders MUST support two modes: fail-safe for production and strict for testing/CI.

#### Scenario: Production fail-safe mode
```python
Given the system is in production environment
When load_backend_config_from_env(fail_safe=True) is called
And a required environment variable is missing
Then the loader MUST use a safe default value
And the loader MUST log a warning about the missing variable
And the loader MUST return a valid config object
And the application MUST start successfully
```

#### Scenario: Testing strict mode
```python
Given the system is in testing/CI environment
When load_backend_config_from_env(fail_safe=False) is called
And a required environment variable is missing
Then the loader MUST raise a ValueError
And the error message MUST identify the missing variable
And the application MUST fail to start
```

#### Scenario: Invalid environment variable value
```python
Given an environment variable has an invalid value
When loading in strict mode (fail_safe=False)
Then the loader MUST raise a TypeError or ValueError
And the error MUST describe the validation failure

When loading in fail-safe mode (fail_safe=True)
Then the loader MUST log a warning
And the loader MUST use the default value
And the loader MUST return a valid config
```

### Requirement: Default Configuration Values

The system MUST provide complete default configurations that allow the application to run without any environment variables set.

#### Scenario: Backend default configuration
```python
Given no environment variables are set
When load_backend_config_from_env(fail_safe=True) is called
Then it returns DEFAULT_CONFIG with safe fallback values
And DEFAULT_CONFIG.api.upload_data_url points to localhost
And all path fields point to temporary directories
And the configuration is valid and complete
```

#### Scenario: Zero-configuration development mode
```python
Given a developer clones the repository
When they run the application without setting env vars
Then the application starts with DEFAULT_CONFIG
And all services use safe localhost endpoints
And temporary directories are created automatically
```

### Requirement: Configuration Documentation

Every configuration field MUST be self-documenting with type hints, docstrings, and clear boundaries.

#### Scenario: Configuration field documentation
```python
Given a configuration dataclass is defined
When inspecting any field
Then the field MUST have a type annotation
And the dataclass MUST have a docstring
And complex fields MUST have inline comments
And the docstring MUST describe purpose and valid values
```

## MODIFIED Requirements

None. This is a new capability being added to the system.

## REMOVED Requirements

None. This change is purely additive and maintains backward compatibility.
