# Spec Delta: Task Execution

## ADDED Requirements

### Requirement: Task Path Parameter Injection
Task functions SHALL accept path configuration as explicit parameters rather than reading from environment variables, enabling environment-independent execution.

#### Scenario: Path parameters provided by dispatcher
- **WHEN** backend service dispatches `task_pipeline_inference` with path parameters
- **THEN** task SHALL use paths from `func_params` dictionary
- **AND** task SHALL NOT read paths from environment variables when parameters provided

#### Scenario: Environment fallback for backward compatibility
- **WHEN** task is dispatched without path parameters
- **THEN** task SHALL fall back to reading paths from environment variables
- **AND** task SHALL log warning about using fallback mechanism
- **AND** task SHALL raise clear error if neither parameters nor environment variables available

#### Scenario: Dual deployment with shared GPU worker
- **WHEN** Production backend dispatches task with `/prod` paths
- **AND** Testing backend dispatches task with `/test` paths
- **AND** both tasks execute on same GPU worker
- **THEN** Production task SHALL execute in `/prod` directory context
- **AND** Testing task SHALL execute in `/test` directory context
- **AND** paths SHALL NOT conflict or interfere between environments

### Requirement: Path Configuration Helper
The system SHALL provide centralized configuration helper for retrieving task execution paths.

#### Scenario: Retrieve paths from current environment
- **WHEN** backend service calls `get_task_execution_paths()`
- **THEN** helper SHALL return dict with `path_process`, `path_json`, `path_log`
- **AND** values SHALL be read from environment variables
- **AND** all paths SHALL be validated as absolute paths

#### Scenario: Path validation failures
- **WHEN** required path is not configured in environment
- **THEN** helper SHALL raise `ValueError` with clear message indicating missing path
- **WHEN** path is relative instead of absolute
- **THEN** helper SHALL raise `ValueError` with message requiring absolute path

#### Scenario: Override paths for testing
- **WHEN** helper is called with override dict
- **THEN** override values SHALL take precedence over environment variables
- **AND** paths SHALL still be validated

### Requirement: Task Parameter Schema
Task dispatcher SHALL include path configuration in task parameters dictionary.

#### Scenario: Complete path parameters for pipeline inference
- **WHEN** dispatching `task_pipeline_inference`
- **THEN** `func_params` SHALL include `path_process` (base directory)
- **AND** `func_params` SHALL include `path_json` (JSON output directory)
- **AND** `func_params` SHALL include `path_log` (log file directory)
- **AND** all paths SHALL be absolute paths

#### Scenario: Path parameters for subprocess inference
- **WHEN** dispatching `task_subprocess_inference`
- **THEN** `func_params` SHALL include `path_process` (base directory)
- **AND** path SHALL be absolute path

## MODIFIED Requirements

### Requirement: Task Environment Independence
Task worker processes SHALL execute tasks using only configuration provided in task parameters, achieving pure function behavior.

#### Scenario: Worker without environment configuration
- **WHEN** GPU worker starts without `.env` file
- **AND** task is dispatched with complete path parameters
- **THEN** task SHALL execute successfully using provided paths
- **AND** task SHALL NOT fail due to missing environment variables

#### Scenario: Task execution determinism
- **WHEN** same task parameters provided multiple times
- **THEN** task SHALL execute identically each time
- **AND** results SHALL be deterministic regardless of worker environment

#### Scenario: Task parameter validation at execution time
- **WHEN** task begins execution
- **THEN** task SHALL validate all required path parameters are present
- **AND** task SHALL validate paths are accessible and writable
- **AND** task SHALL create missing directories if parent exists and is writable
- **AND** task SHALL fail fast with clear error if validation fails

## Design Notes

### Parameter Naming Convention
- Environment variables: Uppercase with underscores (e.g., `PATH_PROCESS`)
- Function parameters: Lowercase with underscores (e.g., `path_process`)
- Dictionary keys: Lowercase with underscores in quotes (e.g., `'path_process'`)

### Validation Strategy
- **Dispatch Time**: Validate path configuration before task dispatch (fast failure)
- **Execution Time**: Re-validate paths before execution (defense in depth)

### Migration Path
- **Phase 1**: Support parameter + environment fallback (current)
- **Phase 2**: Warn on environment fallback for backend services
- **Phase 3**: Deprecate environment fallback for backend (keep for CLI)

### Related Changes
- `parameterize-upload-api-url`: Establishes pattern for parameter injection
- `integrate-dual-deployment-gpu-solution`: Primary use case for this capability
