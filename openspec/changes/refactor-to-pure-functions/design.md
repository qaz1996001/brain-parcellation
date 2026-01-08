# Design: Refactor to Pure Functions

## Architectural Vision

Transform the codebase from imperative, stateful architecture to functional, declarative architecture through:
1. **Side Effect Isolation**: All I/O and state at system boundaries
2. **Dependency Injection**: Explicit dependencies via constructor parameters
3. **Immutable Configuration**: Config objects are frozen dataclasses
4. **Zero-Risk Migration**: Absolute backward compatibility guarantees

## Design Principles

### Knuth's Precision Principle
- Every configuration has explicit type, boundaries, validation
- Configuration is self-documenting
- Correctness first, then performance

### Linus's Data Structure Priority
- Good data structures make code naturally simple
- Eliminate special cases, unify handling logic
- Bottom-up construction from foundation components

### Linus's Backward Compatibility
- **"We do not break userspace"** - external interfaces forever stable
- Internal implementation can evolve freely, external behavior stays consistent
- Use Adapter Pattern to maintain old interfaces, route internally to new implementation
- Enforce backward compatibility, eliminate upgrade risks

## Architecture Patterns

### 1. Configuration Centralization Pattern

**Problem**: 350+ scattered `os.getenv()` calls, no type safety, runtime errors

**Solution**: Immutable configuration dataclasses with centralized loading

```python
@dataclass(frozen=True)
class BackendConfig:
    api: APIConfig
    paths: PathConfig
    database: DatabaseConfig

def load_backend_config_from_env(fail_safe: bool = True) -> BackendConfig:
    """
    Single source of truth for environment loading

    Args:
        fail_safe: True (production) = never fails, uses defaults
                   False (CI/testing) = strict validation, fails fast
    """
    # All os.getenv() calls concentrated here
    # Type conversion and validation in one place
    # Production vs Testing behavior controlled
```

**Benefits**:
- 350+ → 2 environment reads (99.4% reduction)
- Type safety via dataclass annotations
- Fail-safe production, strict testing
- Single source of truth for configuration

### 2. Adapter Pattern for Backward Compatibility

**Problem**: Existing code instantiates `SyncService()` without parameters

**Solution**: Maintain old signature forever, route to new implementation

```python
# NEW: Pure function implementation (internal)
class SyncServiceV2(BaseService):
    def __init__(self, config: BackendConfig):
        super().__init__(config)

    async def run_sync_inference_task(self, data):
        # ✅ Pure function: all config from self.config
        upload_url = self.api.upload_data_url  # No os.getenv()

# OLD: Adapter maintaining eternal compatibility (external)
class SyncService:
    """Backward compatibility adapter (永久相容層)"""
    _v2_instance: Optional[SyncServiceV2] = None

    def __init__(self):
        # ✅ OLD signature preserved forever
        if not self._v2_instance:
            config = load_backend_config_from_env()
            self._v2_instance = SyncServiceV2(config)
        self._impl = self._v2_instance

    async def run_sync_inference_task(self, data):
        # Adapter: OLD signature → NEW implementation
        return await self._impl.run_sync_inference_task(data)
```

**Guarantees**:
- ✅ Old code works forever unchanged
- ✅ Zero migration cost for existing callers
- ✅ New code can use pure function directly
- ✅ Both patterns coexist indefinitely

### 3. Three-Phase Migration Pattern

**Phase A: Foundation** (Pure Addition, Zero Changes)
- Create config dataclasses and loaders
- No modifications to existing code
- Git tag: `v2.0.0-phase-a-foundation`

**Phase B: Parallel Systems** (Coexistence)
- Add V2 implementations alongside V1
- Adapter Pattern maintains V1 compatibility
- Both old and new patterns work identically
- Git tag: `v2.0.0-phase-b-parallel`

**Phase C: Gradual Migration** (Optional)
- Migrate one pipeline at a time
- Contract tests validate each migration
- Old patterns remain available forever
- Git tags: `v2.0.0-phase-c-pipeline-N`

**Never**: Forced migration or breaking changes

### 4. Fail-Safe Configuration Loading

**Production Environment**:
```python
# Production: NEVER fails, always provides working config
config = load_backend_config_from_env(fail_safe=True)
# ✅ Missing env vars → use safe defaults
# ✅ Invalid values → log warning, use defaults
# ✅ System always starts successfully
```

**Testing/CI Environment**:
```python
# Testing: STRICT validation, fail fast
config = load_backend_config_from_env(fail_safe=False)
# ✅ Missing required env vars → raises ValueError
# ✅ Invalid values → raises TypeError
# ✅ Configuration errors caught immediately
```

**Benefits**:
- Production: High availability, graceful degradation
- Testing: Early error detection, strict validation
- Clear separation of concerns
- Environment-aware behavior

### 5. Contract Testing for Behavioral Equivalence

**Problem**: How to prove OLD ≡ NEW implementation?

**Solution**: Automated contract tests with property-based testing

```python
class TestBackwardCompatibilityContract:
    """Contract: OLD implementation ≡ NEW implementation"""

    async def test_sync_service_contract(self, comprehensive_test_data):
        # OLD implementation
        old_service = SyncService()
        old_result = await old_service.run_sync_inference_task(test_data)

        # NEW implementation
        config = load_backend_config_from_env()
        new_service = SyncServiceV2(config)
        new_result = await new_service.run_sync_inference_task(test_data)

        # ✅ Mathematical proof of identical behavior
        assert old_result == new_result
        assert old_service._impl is new_service  # Same implementation

    @given(st.lists(st.builds(TaskRequest)))
    def test_pipeline_contract_property(self, requests):
        """Property-based testing: ∀ inputs, OLD(x) = NEW(x)"""
        # Hypothesis generates exhaustive test cases
        # Proves equivalence across all edge cases
```

**Benefits**:
- Automated mathematical proof, not manual testing
- Property-based testing covers edge cases
- CI gates prevent behavioral drift
- Confidence in zero-risk migration

### 6. Feature Flags for Runtime Rollback

**Problem**: Need instant rollback without redeployment

**Solution**: Runtime feature flags for system selection

```python
class FeatureFlags:
    @staticmethod
    def use_new_config_system() -> bool:
        """
        Environment:
            USE_NEW_CONFIG=true  → NEW system (default)
            USE_NEW_CONFIG=false → OLD system (instant rollback)
        """
        return os.getenv("USE_NEW_CONFIG", "true").lower() == "true"

# Service entry point
if FeatureFlags.use_new_config_system():
    service = SyncServiceV2(load_backend_config_from_env())
else:
    service = SyncService()  # Instant rollback to old implementation
```

**Rollback procedure** (< 10 seconds):
```bash
# Instant rollback without code changes
export USE_NEW_CONFIG=false
systemctl restart backend
# ✅ System immediately uses old implementation
```

**Benefits**:
- < 10 second rollback time
- Zero code changes required
- A/B testing capability
- Gradual rollout control

## Trade-offs and Decisions

### Decision 1: Adapter Pattern vs Direct Migration

**Options**:
- A) Adapter Pattern (chosen)
- B) Update all call sites directly
- C) Keep both systems separate

**Rationale**:
- ✅ Adapter: Zero breaking changes, infinite backward compatibility
- ❌ Direct: High risk, requires coordinated changes across 100+ files
- ❌ Separate: Code duplication, maintenance burden

**Trade-off**: Small adapter overhead vs zero migration risk
**Choice**: Zero risk wins (Linus principle)

### Decision 2: Fail-Safe vs Fail-Fast for Production

**Options**:
- A) Fail-safe production with defaults (chosen)
- B) Fail-fast everywhere
- C) Silent failures

**Rationale**:
- ✅ Fail-safe: High availability, graceful degradation
- ❌ Fail-fast production: Service downtime on minor config issues
- ❌ Silent: Hidden bugs, debugging nightmares

**Trade-off**: Potential misconfigurations vs service availability
**Choice**: Availability wins for production, strictness for testing

### Decision 3: Three-Phase vs Big-Bang Migration

**Options**:
- A) Three-phase gradual migration (chosen)
- B) Big-bang rewrite
- C) Feature branch merge

**Rationale**:
- ✅ Three-phase: Incremental verification, continuous integration
- ❌ Big-bang: High risk, long integration period, rollback difficult
- ❌ Feature branch: Merge conflicts, divergent codebase

**Trade-off**: Longer timeline vs zero integration risk
**Choice**: Zero risk wins (proven by Linux Kernel model)

## Success Criteria

### Functional Requirements
- ✅ All existing tests pass with zero modifications
- ✅ Contract tests prove OLD ≡ NEW for all services
- ✅ Configuration errors detected at startup, not runtime
- ✅ Feature flag rollback completes in < 10 seconds

### Non-Functional Requirements
- ✅ Environment variable reads: 350+ → 2 (99.4% reduction)
- ✅ Side effect functions: 150+ → 0 (100% elimination)
- ✅ Test isolation: Complete (mock config, not environment)
- ✅ Backward compatibility: Absolute (old code works forever)

### Quality Gates
- ✅ `pytest` with 100% contract test coverage
- ✅ `mypy --strict` passes for all new code
- ✅ `openspec validate` passes with zero warnings
- ✅ Git tag at each phase for rollback points

## Implementation Sequence

### Phase A: Foundation (Week 1-2)
1. Create `backend/app/config/` with dataclasses
2. Implement `load_backend_config_from_env()`
3. Create `code_ai/config/` with CodeAIConfig
4. Contract test framework setup
5. Git tag: `v2.0.0-phase-a-foundation`

### Phase B: Parallel Systems (Week 3-4)
1. Implement `SyncServiceV2` with dependency injection
2. Create `SyncService` adapter
3. Contract tests: `SyncService ≡ SyncServiceV2`
4. Feature flags implementation
5. Git tag: `v2.0.0-phase-b-parallel`

### Phase C: Pipeline Migration (Week 5-10)
1. Migrate `CMBPipeline` with dual-mode support
2. Contract tests: `old main() ≡ new main(config)`
3. Repeat for remaining 6 pipelines
4. Git tags: `v2.0.0-phase-c-pipeline-N`

### Phase D: Validation (Week 11-12)
1. Full integration testing
2. Performance benchmarking
3. Documentation updates
4. Git tag: `v2.0.0-complete`

## Risks and Mitigation

**This section intentionally minimal** - proper design eliminates risks through programming techniques, not mitigation strategies.

**Zero-risk guarantees**:
- ✅ Adapter Pattern: Old code works forever
- ✅ Contract Testing: Automated proof of equivalence
- ✅ Feature Flags: Instant rollback capability
- ✅ Three-Phase Migration: Incremental verification
- ✅ Fail-Safe Production: Never crashes on config issues

**Defense-in-depth rollback**:
- Layer 1: Feature flags (< 10 seconds)
- Layer 2: Git revert (5-10 minutes)
- Layer 3: Git tag reset (10-15 minutes)

**This is engineering maturity, not failure expectation** - following Linux Kernel, Google SRE, AWS standards.

## References

- **Primary**: `openspec/changes/add-environment-support/PURE_FUNCTION_REFACTORING_PLAN.md`
- **Principles**: `openspec/changes/add-environment-support/KNUTH_LINUS_DESIGN.md`
- **Call Chain**: `openspec/changes/add-environment-support/CALL_CHAIN_ANALYSIS.md`
- **Python PEP 387**: Backwards Compatibility Policy
- **Martin Fowler**: Refactoring - Improving the Design of Existing Code
- **Robert Martin**: Clean Architecture
