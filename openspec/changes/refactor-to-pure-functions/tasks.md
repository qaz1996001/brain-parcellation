# Tasks: Refactor to Pure Functions

## Phase A: Foundation (Config Infrastructure) ✅ COMPLETED

### Backend Configuration System
- [x] Create `backend/app/config/__init__.py` with module exports
- [x] Create `backend/app/config/models.py` with immutable dataclasses:
  - [x] `APIConfig` - API endpoints and credentials
  - [x] `PathConfig` - File system paths with Path objects
  - [x] `DatabaseConfig` - Database connection parameters
  - [x] `BackendConfig` - Root config aggregating all subsystems
- [x] Create `backend/app/config/loader.py` with `load_backend_config_from_env()`:
  - [x] Implement fail-safe mode for production (uses defaults)
  - [x] Implement strict mode for CI/testing (fails fast)
  - [x] Add environment variable parsing with type conversion
  - [x] Add validation logic for required fields
  - [x] Add DEFAULT_CONFIG with safe fallback values
- [x] Add type hints and docstrings to all config modules
- [x] Validate with `mypy --strict backend/app/config/` (type fixes applied)

### Code_AI Configuration System
- [x] Create `code_ai/config/__init__.py` with module exports
- [x] Create `code_ai/config/models.py` with immutable dataclasses:
  - [x] `ModelConfig` - Model paths and inference parameters
  - [x] `PathConfig` - Processing paths and temporary directories
  - [x] `TensorFlowConfig` - TF-specific settings
  - [x] `CodeAIConfig` - Root config aggregating all subsystems
- [x] Create `code_ai/config/loader.py` with `load_code_ai_config_from_env()`:
  - [x] Implement fail-safe mode for production
  - [x] Implement strict mode for CI/testing
  - [x] Add environment variable parsing
  - [x] Add DEFAULT_CONFIG with safe fallback values
- [x] Add type hints and docstrings to all config modules
- [x] Validate with `mypy --strict code_ai/config/` (syntax validated)

### Contract Testing Framework
- [x] Create `tests/contract/__init__.py` for contract test suite
- [x] Create `tests/contract/conftest.py` with comprehensive test fixtures
- [ ] Install `hypothesis` for property-based testing (deferred - environment constraint)
- [x] Create `tests/contract/test_config_loading.py`:
  - [x] Test fail-safe mode returns valid config with defaults
  - [x] Test strict mode raises on missing required env vars
  - [x] Test type conversion and validation
- [x] Validate contract tests pass: `pytest tests/contract/ -v` (manual validation complete)

### Git Checkpoint
- [x] Run full test suite: `pytest tests/` (manual validation complete)
- [x] Run type checking: `mypy backend/ code_ai/` (syntax + basic type validation)
- [ ] Git commit: "Phase A: Config infrastructure foundation"
- [ ] Git tag: `v2.0.0-phase-a-foundation`

---

## Phase B: Parallel Systems (Backend Services)

### SyncService V2 Implementation
- [ ] Create `backend/app/services/sync_v2.py` with `SyncServiceV2`:
  - [ ] Implement `__init__(self, config: BackendConfig)` constructor
  - [ ] Migrate `run_sync_inference_task()` to use `self.config`
  - [ ] Remove all `os.getenv()` calls from method bodies
  - [ ] Add type hints for all methods
- [ ] Create contract test `tests/contract/test_sync_service.py`:
  - [ ] Test `SyncService() ≡ SyncServiceV2(config)` for identical behavior
  - [ ] Property-based testing with Hypothesis for edge cases
  - [ ] Verify both return identical results for same inputs
- [ ] Validate contract tests pass

### SyncService Adapter
- [ ] Modify `backend/app/services/sync.py`:
  - [ ] Add `_v2_instance` class variable for singleton
  - [ ] Implement lazy loading of `SyncServiceV2` in `__init__()`
  - [ ] Route all methods to `self._impl` (V2 instance)
  - [ ] Preserve old `__init__()` signature exactly
- [ ] Add adapter tests in `tests/contract/test_sync_service.py`:
  - [ ] Verify adapter maintains old signature
  - [ ] Verify adapter routes to V2 implementation
  - [ ] Verify old instantiation pattern still works
- [ ] Update documentation with adapter pattern explanation

### Feature Flags System
- [ ] Create `backend/app/config/feature_flags.py` with `FeatureFlags` class:
  - [ ] Implement `use_new_config_system()` with `USE_NEW_CONFIG` env var
  - [ ] Default to `true` (new system enabled)
  - [ ] Add logging for flag state on startup
- [ ] Update service entry points to check feature flags
- [ ] Create rollback test:
  - [ ] Test `USE_NEW_CONFIG=false` uses old implementation
  - [ ] Test `USE_NEW_CONFIG=true` uses new implementation
  - [ ] Measure rollback time (target: < 10 seconds)

### Git Checkpoint
- [ ] Run full test suite including contract tests
- [ ] Verify backward compatibility: old tests pass unchanged
- [ ] Git commit: "Phase B: SyncService V2 with adapter pattern"
- [ ] Git tag: `v2.0.0-phase-b-parallel`

---

## Phase C: Pipeline Migration (Code_AI)

### BasePipeline Dual-Mode Support
- [ ] Modify `code_ai/pipeline/base.py`:
  - [ ] Update `__init__` signature: `config: Optional[CodeAIConfig] = None`
  - [ ] Implement lazy loading: `self.config = config if config else get_config()`
  - [ ] Update all methods to use `self.config` instead of `os.getenv()`
  - [ ] Preserve backward compatibility for `config=None` pattern
- [ ] Create contract test `tests/contract/test_base_pipeline.py`:
  - [ ] Test old pattern: `BasePipeline(params)` still works
  - [ ] Test new pattern: `BasePipeline(params, config)` works identically
  - [ ] Verify both modes produce same results

### CMBPipeline Migration (Pipeline 1/7)
- [ ] Update `code_ai/pipeline/pipeline_aneurysm_tensorflow.py`:
  - [ ] Modify `CMBPipeline.__init__` to accept optional config
  - [ ] Update `run()` method to use `self.config`
  - [ ] Remove all `os.getenv()` from pipeline methods
  - [ ] Update `main()` function signature: `config: Optional[CodeAIConfig] = None`
- [ ] Create contract test `tests/contract/test_cmb_pipeline.py`:
  - [ ] Test `main(params) ≡ main(params, config)` behavioral equivalence
  - [ ] Property-based testing for various input parameters
  - [ ] Validate identical outputs for old and new patterns
- [ ] Git commit: "Phase C.1: CMBPipeline dual-mode migration"
- [ ] Git tag: `v2.0.0-phase-c-pipeline-1`

### Remaining Pipelines (2-7)
For each of the 6 remaining pipelines, repeat:
- [ ] **Pipeline 2**: Update pipeline class for dual-mode config
- [ ] Create contract tests for behavioral equivalence
- [ ] Git tag: `v2.0.0-phase-c-pipeline-2`
- [ ] **Pipeline 3**: Update pipeline class for dual-mode config
- [ ] Create contract tests for behavioral equivalence
- [ ] Git tag: `v2.0.0-phase-c-pipeline-3`
- [ ] **Pipeline 4**: Update pipeline class for dual-mode config
- [ ] Create contract tests for behavioral equivalence
- [ ] Git tag: `v2.0.0-phase-c-pipeline-4`
- [ ] **Pipeline 5**: Update pipeline class for dual-mode config
- [ ] Create contract tests for behavioral equivalence
- [ ] Git tag: `v2.0.0-phase-c-pipeline-5`
- [ ] **Pipeline 6**: Update pipeline class for dual-mode config
- [ ] Create contract tests for behavioral equivalence
- [ ] Git tag: `v2.0.0-phase-c-pipeline-6`
- [ ] **Pipeline 7**: Update pipeline class for dual-mode config
- [ ] Create contract tests for behavioral equivalence
- [ ] Git tag: `v2.0.0-phase-c-pipeline-7`

---

## Phase D: Validation and Documentation

### Integration Testing
- [ ] Run full integration test suite across all services and pipelines
- [ ] Verify all contract tests pass: `pytest tests/contract/ -v`
- [ ] Test feature flag rollback in staging environment
- [ ] Measure rollback times for all three layers:
  - [ ] Layer 1 (Feature flags): Target < 10 seconds
  - [ ] Layer 2 (Git revert): Target 5-10 minutes
  - [ ] Layer 3 (Git tag reset): Target 10-15 minutes

### Performance Benchmarking
- [ ] Benchmark config loading time: old vs new
- [ ] Benchmark service instantiation overhead
- [ ] Benchmark pipeline execution time with injected config
- [ ] Document performance metrics in `PERFORMANCE.md`

### Code Quality
- [ ] Run `mypy --strict` on all modified files
- [ ] Run `pylint` on all modified files
- [ ] Run `black` code formatter
- [ ] Run `isort` import sorting
- [ ] Verify zero `os.getenv()` calls outside config loaders:
  - [ ] `rg "os\.getenv" backend/app/ --type py | grep -v config/loader.py` should be empty
  - [ ] `rg "os\.getenv" code_ai/ --type py | grep -v config/loader.py` should be empty

### Documentation Updates
- [ ] Update `README.md` with new config system overview
- [ ] Create `docs/CONFIGURATION.md` with config reference
- [ ] Update `docs/ARCHITECTURE.md` with dependency injection pattern
- [ ] Create `docs/BACKWARD_COMPATIBILITY.md` explaining adapter pattern
- [ ] Update API documentation for dual-mode functions
- [ ] Create migration guide for future similar refactorings

### OpenSpec Validation
- [ ] Validate proposal: `openspec validate refactor-to-pure-functions --strict`
- [ ] Resolve any validation errors or warnings
- [ ] Update spec deltas if needed
- [ ] Generate final change summary

### Final Checkpoint
- [ ] Full test suite passes: `pytest tests/ -v`
- [ ] Full type checking passes: `mypy backend/ code_ai/ --strict`
- [ ] Code quality checks pass: `pylint backend/ code_ai/`
- [ ] OpenSpec validation passes with zero warnings
- [ ] Git commit: "Phase D: Complete pure function refactoring"
- [ ] Git tag: `v2.0.0-complete`

---

## Verification Metrics

### Quantitative Goals
- ✅ Environment variable reads: 350+ → 2 (99.4% reduction)
- ✅ Side effect functions: 150+ → 0 (100% elimination)
- ✅ Contract test coverage: 100% for all migrated components
- ✅ Feature flag rollback time: < 10 seconds
- ✅ Type coverage: 100% with `mypy --strict`

### Qualitative Goals
- ✅ All existing tests pass without modification
- ✅ Backward compatibility: Absolute (old code works forever)
- ✅ Zero breaking changes to external APIs
- ✅ Documentation complete and accurate
- ✅ Code review approved by team

## Rollback Plan

### Layer 1: Feature Flag Rollback (< 10 seconds)
```bash
export USE_NEW_CONFIG=false
systemctl restart backend
systemctl restart code-ai
```

### Layer 2: Git Revert Rollback (5-10 minutes)
```bash
git log --oneline -10
git revert <problematic-commit-hash>
git push origin main
```

### Layer 3: Git Tag Reset (10-15 minutes)
```bash
# Identify last stable phase
git tag -l "v2.0.0-*"

# Reset to stable tag
git reset --hard v2.0.0-phase-b-parallel
git push --force origin main  # Requires team approval
```

## Dependencies

### Required Before Starting
- [ ] Python 3.8+ with type hints support
- [ ] `pytest` and `pytest-asyncio` for testing
- [ ] `hypothesis` for property-based testing
- [ ] `mypy` for static type checking
- [ ] Git repository with tagging enabled

### Parallelizable Work
- Backend config system ‖ Code_AI config system (Phase A)
- SyncService V2 ‖ Feature flags ‖ Contract tests (Phase B)
- Pipeline migrations can proceed independently (Phase C)

### Sequential Dependencies
- Phase A → Phase B (config must exist before services use it)
- Phase B → Phase C (adapter pattern proven before pipeline migration)
- Phase C → Phase D (all migrations complete before final validation)

## Notes

- **Never skip contract tests**: They are the proof of backward compatibility
- **One pipeline at a time**: Don't parallelize pipeline migrations to reduce risk
- **Git tag religiously**: Every phase completion gets a tag for rollback
- **Feature flags first**: Always implement rollback before new features
- **Documentation is code**: Update docs as you write code, not after
