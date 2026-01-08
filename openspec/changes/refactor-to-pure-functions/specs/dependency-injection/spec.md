# Spec: Dependency Injection

## ADDED Requirements

### Requirement: Constructor-Based Dependency Injection

All service classes MUST receive their configuration dependencies through constructor parameters, not by reading environment variables directly.

#### Scenario: Service V2 implementation with dependency injection
```python
Given a new service implementation (V2)
When the service is instantiated
Then it MUST accept a config parameter in __init__
And the config parameter MUST have a type annotation
And the service MUST NOT call os.getenv() anywhere
And all configuration MUST come from self.config
```

#### Scenario: Pure function service methods
```python
Given a service with dependency-injected configuration
When any service method is called
Then the method MUST be a pure function
And the method MUST use only self.config for configuration
And the method MUST NOT access global state
And the method MUST NOT read environment variables
```

### Requirement: Dual-Mode Pipeline Support

Pipeline classes MUST support both legacy (no config parameter) and modern (with config parameter) instantiation patterns.

#### Scenario: Legacy pipeline instantiation
```python
Given a pipeline class with dual-mode support
When instantiated with pipeline = Pipeline(func_params)
Then the pipeline MUST lazy-load configuration
And the configuration MUST come from get_config()
And the pipeline MUST function identically to old behavior
```

#### Scenario: Modern pipeline instantiation
```python
Given a pipeline class with dual-mode support
When instantiated with pipeline = Pipeline(func_params, config=config)
Then the pipeline MUST use the injected config
And the pipeline MUST NOT call get_config()
And the pipeline MUST function identically to legacy behavior
```

#### Scenario: Pipeline dual-mode equivalence
```python
Given test data for a pipeline
When running legacy_pipeline = Pipeline(data)
And running modern_pipeline = Pipeline(data, config=load_config())
Then legacy_pipeline.run() == modern_pipeline.run()
And both produce identical outputs
And both have identical side effects
```

### Requirement: Explicit Dependencies

Service and pipeline classes MUST make all their dependencies explicit through constructor signatures and type annotations.

#### Scenario: Service dependency declaration
```python
Given a service class definition
When inspecting the __init__ signature
Then all dependencies MUST be explicit parameters
And all parameters MUST have type annotations
And no hidden dependencies through global state
And no hidden dependencies through environment variables
```

#### Scenario: Configuration as first-class dependency
```python
Given any service or pipeline class
When the class needs configuration
Then config MUST be a constructor parameter
And config MUST have a type annotation (e.g., BackendConfig)
And config MUST be stored as self.config
And all methods MUST use self.config exclusively
```

### Requirement: Testability Through Dependency Injection

All services and pipelines MUST be fully testable by injecting mock or test configurations.

#### Scenario: Service testing with mock configuration
```python
Given a service that uses dependency injection
When writing unit tests for the service
Then tests MUST create a mock configuration object
And tests MUST inject the mock config into the service
And tests MUST NOT need to modify environment variables
And tests MUST run in complete isolation
```

#### Scenario: Pipeline testing with test configuration
```python
Given a pipeline that supports config injection
When writing integration tests for the pipeline
Then tests MUST create a test configuration
And tests MUST inject the test config into the pipeline
And tests MUST use temporary paths and test endpoints
And tests MUST NOT affect production configuration
```

## MODIFIED Requirements

None. This is a new architectural pattern being introduced.

## REMOVED Requirements

None. Legacy instantiation patterns remain supported for backward compatibility.
