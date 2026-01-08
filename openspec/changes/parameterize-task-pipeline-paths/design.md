# Design: Parameterize Task Pipeline Environment Paths

## Context

### Problem Domain
We have a distributed task execution system where:
- **Backend Services** (Production, Testing) dispatch tasks to queues
- **GPU Workers** consume tasks from shared queues and execute inference
- **Current Issue**: Workers read path configuration from their local `.env` files, preventing environment-specific path routing

### Stakeholders
- **DevOps**: Want to run single GPU worker serving multiple environments
- **Backend Developers**: Need to specify environment-specific paths when dispatching tasks
- **QA/Testing**: Need Testing environment to use different paths than Production

### Constraints
- Must maintain backward compatibility for existing deployments
- Cannot break running tasks during migration
- Should follow patterns established in `parameterize-upload-api-url`
- Must work with existing RabbitMQ/Redis infrastructure

## Goals / Non-Goals

### Goals
1. **Pure Function Design**: Transform task functions to accept all configuration as parameters
2. **Environment Decoupling**: Enable backends to specify path configuration independent of worker environment
3. **Backward Compatibility**: Maintain fallback to environment variables during migration period
4. **Dual Deployment**: Enable single GPU worker to serve multiple backend environments with different path configurations

### Non-Goals
- **Not**: Changing task queue infrastructure (RabbitMQ/Redis)
- **Not**: Modifying path directory structures or conventions
- **Not**: Adding new path validation beyond basic existence checks
- **Not**: Creating new configuration file formats (continue using .env)

## Decisions

### Decision 1: Parameter Injection Pattern

**What**: Use dependency injection pattern - task functions receive configuration as explicit parameters rather than reading from environment.

**Why** (Martin Fowler Principles):
1. **Explicit Dependencies**: Parameters make dependencies visible in function signature
2. **Testability**: Can test with different configurations without modifying environment
3. **Composability**: Pure functions can be composed and reused in different contexts
4. **Separation of Concerns**: Configuration logic (dispatcher) separated from execution logic (worker)

**How**:
```python
# Before: Implicit dependency on environment
def task_pipeline_inference(func_params: Dict):
    path_process = os.getenv("PATH_PROCESS")  # Hidden dependency

# After: Explicit dependency via parameters
def task_pipeline_inference(func_params: Dict):
    path_process = func_params.get('path_process')  # Explicit parameter
    if path_process is None:
        path_process = os.getenv("PATH_PROCESS")  # Fallback
```

**Alternatives Considered**:
- **Alternative 1**: Create environment-specific task queues (rejected - infrastructure complexity)
- **Alternative 2**: Use task headers for configuration (rejected - not supported by current queue system)
- **Alternative 3**: Configuration service/database (rejected - over-engineering for current scale)

### Decision 2: Configuration Helper Pattern

**What**: Create centralized configuration helper following Single Responsibility Principle.

**Why**:
1. **Don't Repeat Yourself (DRY)**: Single source of truth for path retrieval logic
2. **Encapsulation**: Hide environment variable reading logic
3. **Testability**: Can mock helper for testing
4. **Consistency**: All callers use same configuration retrieval mechanism

**How**:
```python
# backend/app/config/task_paths.py
def get_task_execution_paths(override: Optional[Dict] = None) -> Dict[str, str]:
    """Get path configuration for task parameter injection.

    Args:
        override: Optional path overrides for testing

    Returns:
        Dict with path_process, path_json, path_log

    Raises:
        ValueError: If required paths not configured
    """
    paths = override or {}

    result = {
        'path_process': paths.get('path_process') or os.getenv("PATH_PROCESS"),
        'path_json': paths.get('path_json') or os.getenv("PATH_JSON"),
        'path_log': paths.get('path_log') or os.getenv("PATH_LOG"),
    }

    # Validate required paths
    for key, value in result.items():
        if value is None:
            raise ValueError(f"{key} must be configured")
        if not os.path.isabs(value):
            raise ValueError(f"{key} must be absolute path, got: {value}")

    return result
```

**Pattern**: Similar to existing `get_upload_data_api_url()` for consistency.

### Decision 3: Backward Compatibility Strategy

**What**: Support environment variable fallback indefinitely for CLI scripts, with gradual deprecation path for backend services.

**Why**:
- **CLI Scripts**: Developer convenience - can run without explicit parameters
- **Backend Services**: Explicit > Implicit - should use parameter injection
- **Migration Path**: Allows gradual rollout without breaking existing deployments

**How**:
```python
# Phase 1 (Current): Support both, no warnings
path = func_params.get('path_process') or os.getenv("PATH_PROCESS")

# Phase 2 (Future): Warn on environment fallback for backend
if 'path_process' not in func_params:
    logger.warning("Using environment fallback for path_process - please migrate to explicit parameters")
    path = os.getenv("PATH_PROCESS")

# Phase 3 (Long-term): Require parameters for backend, keep fallback for CLI
# Detect context and apply different rules
```

### Decision 4: Validation Strategy

**What**: Validate at dispatch time (backend) AND execution time (worker).

**Why**:
- **Fast Failure**: Dispatch-time validation catches misconfiguration early
- **Defense in Depth**: Execution-time validation protects against race conditions (deleted directories, permission changes)
- **Clear Error Messages**: User gets immediate feedback on misconfiguration

**How**:
```python
# Dispatch time (backend/app/config/task_paths.py)
def get_task_execution_paths(...):
    # Validate paths exist and are accessible
    for key, path in result.items():
        if not os.path.exists(path):
            raise ValueError(f"{key} does not exist: {path}")
        if not os.access(path, os.W_OK):
            raise ValueError(f"{key} is not writable: {path}")
    return result

# Execution time (code_ai/task/task_pipeline.py)
def task_pipeline_inference(func_params):
    paths = extract_paths_from_params(func_params)
    # Re-validate and create if needed
    os.makedirs(paths['path_json'], exist_ok=True)
    os.makedirs(paths['path_log'], exist_ok=True)
```

### Decision 5: Parameter Naming Consistency

**What**: Use consistent naming across parameters and environment variables.

**Why**:
- **Predictability**: Developers can easily map parameters to environment variables
- **Maintainability**: Clear 1:1 relationship simplifies debugging
- **Documentation**: Self-documenting code through consistent naming

**Naming Convention**:
- Environment Variable: `PATH_PROCESS` (uppercase, underscore)
- Function Parameter: `path_process` (lowercase, underscore)
- Dict Key: `'path_process'` (lowercase, underscore in quotes)

```python
# Consistent mapping
ENV_VAR_NAME = "PATH_PROCESS"
PARAM_NAME = "path_process"
func_params.get('path_process')  # Matches param name
```

## Architecture

### Current State (Before)

```
┌─────────────────────┐
│ Backend Service     │
│ (Production)        │
│ .env:               │
│   PATH_PROCESS=/prod│
└──────────┬──────────┘
           │ push task_dict
           │ {nifti_path, dicom_path}
           ▼
    ┌─────────────┐
    │ RabbitMQ    │
    │   Queue     │
    └──────┬──────┘
           │
           ▼
┌─────────────────────┐
│ GPU Worker          │
│ .env:               │
│   PATH_PROCESS=/gpu │  ← Worker env determines paths
└─────────────────────┘

Problem: Worker .env overrides backend configuration
```

### Proposed State (After)

```
┌─────────────────────┐
│ Backend Service     │
│ (Production)        │
│ .env:               │
│   PATH_PROCESS=/prod│
│                     │
│ get_task_paths() ───┼──> paths = {
│                     │      path_process: /prod,
└──────────┬──────────┘      path_json: /prod/json,
           │ push task_dict       path_log: /prod/log
           │ {nifti, dicom,    }
           │  path_process,
           │  path_json,
           │  path_log}
           ▼
    ┌─────────────┐
    │ RabbitMQ    │
    │   Queue     │
    └──────┬──────┘
           │
           ▼
┌─────────────────────┐
│ GPU Worker          │
│ (No .env needed)    │  ← Uses params from task_dict
│                     │
│ func_params.get() ──┼──> paths from params
└─────────────────────┘

Solution: Backend configuration flows through parameters
```

### Dual Deployment Flow

```
Production Backend              Testing Backend
  (D:\Task04_git)                (D:\Task04_git_test)
  .env: PATH=/prod              .env: PATH=/test
        │                              │
        │ paths={                      │ paths={
        │   path_process:/prod}        │   path_process:/test}
        │                              │
        └──────────┬──────────────────┬┘
                   │                  │
                   ▼                  ▼
            ┌─────────────────┐
            │   RabbitMQ      │
            │ Shared Queue    │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  GPU Worker     │
            │ (Single Process)│
            │ - Reads paths   │
            │   from params   │
            │ - Executes in   │
            │   specified dir │
            └─────────────────┘

Benefit: One GPU worker serves both environments
```

## Implementation Phases

### Phase 1: Infrastructure (Parallel)
1. Create `backend/app/config/task_paths.py` helper
2. Write unit tests for helper function
3. Add validation logic with error handling

### Phase 2: Task Function Updates (Sequential after Phase 1)
1. Update `task_pipeline_inference` parameter extraction
2. Update `task_subprocess_inference` parameter extraction
3. Maintain environment fallback logic
4. Add inline documentation

### Phase 3: Dispatcher Updates (Parallel after Phase 2)
1. Update `backend/app/sync/service.py`
2. Update `backend/app/listen/service.py`
3. Update `backend/app/study/service.py`
4. Update `code_ai/scheduler/scheduler_check_add_task.py`

### Phase 5: Testing and Validation (Sequential after all)
1. Unit tests for pure functions with different path configurations
2. Integration tests for dual deployment scenario
3. Backward compatibility tests with environment variables
4. End-to-end validation with Production and Testing backends

## Risks / Trade-offs

### Risk 1: Parameter Explosion
**Risk**: Adding more parameters to task functions increases complexity
**Mitigation**:
- Group related parameters into configuration dict
- Consider schema validation (Pydantic) for parameter structure
- Limit to essential configuration only

**Trade-off**: Simplicity (fewer params) vs Explicitness (visible dependencies)
**Decision**: Choose explicitness - makes dependencies clear and testable

### Risk 2: Migration Coordination
**Risk**: Partial migration could cause inconsistent behavior
**Mitigation**:
- Update all callers in single change
- Maintain backward compatibility during transition
- Clear migration documentation

**Trade-off**: Big-bang migration (risky but complete) vs Gradual migration (safer but longer)
**Decision**: Choose gradual with fallback - safer for production

### Risk 3: Performance Impact
**Risk**: Additional parameter passing and validation adds overhead
**Mitigation**:
- Validate only at dispatch time for fast failure
- Cache helper results where appropriate
- Measure performance impact

**Trade-off**: Safety (more validation) vs Performance (less overhead)
**Decision**: Choose safety - path validation is minimal overhead compared to GPU inference

### Risk 4: Configuration Inconsistency
**Risk**: Different backends might specify conflicting or invalid paths
**Mitigation**:
- Require absolute paths for clarity
- Validate path accessibility at dispatch time
- Document path configuration requirements
- Add monitoring for path-related errors

**Trade-off**: Flexibility (any paths) vs Safety (validated paths)
**Decision**: Choose safety - fail fast with clear errors

## Migration Plan

### Step 1: Deploy Infrastructure (Week 1)
- Deploy configuration helper with validation
- No impact on existing functionality
- Establish testing framework

### Step 2: Deploy Task Function Changes (Week 1)
- Update task functions with parameter extraction + fallback
- Backward compatible - existing callers still work
- Monitor for errors in fallback path

### Step 3: Deploy Dispatcher Changes (Week 2)
- Update backend services to pass parameters
- CLI scripts continue using environment fallback
- Validate dual deployment scenario in staging

### Step 4: Production Validation (Week 2)
- Deploy to production with monitoring
- Verify both Production and Testing backends work
- Monitor for configuration errors

### Rollback Plan
If issues occur:
1. **Before Step 3**: Simply don't deploy dispatcher changes
2. **After Step 3**: Revert dispatcher changes, task functions fall back to environment
3. **Emergency**: Remove parameter extraction entirely, use environment only

## Open Questions

### Q1: Should we support relative paths?
**Status**: Resolved
**Decision**: No - require absolute paths for clarity and consistency
**Rationale**: Relative paths create ambiguity about resolution context

### Q2: How to handle path creation failures?
**Status**: Open
**Options**:
- A) Fail fast and notify dispatcher
- B) Retry with exponential backoff
- C) Use fallback default paths
**Recommendation**: Option A - fail fast with clear error message

### Q3: Should we validate path accessibility in helper?
**Status**: Resolved
**Decision**: Yes - validate existence and write permissions
**Rationale**: Fast failure is better than late failure during task execution

### Q4: CLI script parameter vs environment preference?
**Status**: Open
**Options**:
- A) Add CLI arguments for paths (explicit)
- B) Continue using environment only (simple)
- C) Support both with CLI args taking precedence (flexible)
**Recommendation**: Option B for simplicity, can add Option C later if needed

## Success Metrics

### Functional Metrics
- ✅ Single GPU worker successfully serves both Production and Testing
- ✅ All tasks execute in correct path context
- ✅ No environment variable conflicts
- ✅ Backward compatibility maintained

### Quality Metrics
- ✅ Test coverage >90% for new code
- ✅ Zero production incidents from migration
- ✅ All validation errors clear and actionable
- ✅ Documentation complete and accurate

### Performance Metrics
- ✅ Task dispatch latency <10ms increase
- ✅ No measurable impact on task execution time
- ✅ Path validation overhead <1ms

## References

### Martin Fowler Patterns Applied
1. **Dependency Injection**: Explicit parameter passing
2. **Pure Functions**: Deterministic execution with explicit inputs
3. **Separation of Concerns**: Configuration (dispatcher) vs Execution (worker)
4. **Fail Fast**: Validate early, report clearly
5. **Progressive Enhancement**: Backward compatible migration path

### Related Documentation
- OpenSpec: `parameterize-upload-api-url` (similar pattern)
- OpenSpec: `integrate-dual-deployment-gpu-solution` (motivation)
- Code: `backend/app/config/api_urls.py` (reference implementation)

### External References
- Martin Fowler: "Inversion of Control Containers and the Dependency Injection pattern"
- Martin Fowler: "Refactoring - Improving the Design of Existing Code"
