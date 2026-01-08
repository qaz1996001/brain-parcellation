# Tasks: Refactor Service Repository Pattern

## 1. Repository Layer Enhancement

### 1.1 Custom Repository Methods
- [x] 1.1.1 Create `get_studies_ready_for_transfer()` method to encapsulate `get_all_studies_status()` stored procedure call
- [x] 1.1.2 Create `get_series_by_status(status, study_uid=None)` method to unify 4 duplicate query patterns
- [x] 1.1.3 Create `get_studies_pending_completion(status)` method for complex subquery logic
- [x] 1.1.4 Add type hints and docstrings for all new Repository methods
- [ ] 1.1.5 Write unit tests for Repository methods with mock session

### 1.2 Status URL Mapping Data Structure
- [x] 1.2.1 Define `STATUS_URL_MAP: Dict[DCOPStatus, str]` class constant
- [x] 1.2.2 Refactor `get_check_url_by_ope_no()` to use dict lookup
- [ ] 1.2.3 Write tests validating mapping completeness

## 2. Service Layer Refactoring

### 2.1 Replace Direct Session Operations
- [ ] 2.1.1 Refactor `post_ope_no_task()`: Replace loop commit with `create_many()`
- [ ] 2.1.2 Refactor `add_study_new()`: Use `create_many()` for batch insert
- [ ] 2.1.3 Refactor `study_series_nifti_tool()`: Use Repository `create()` method
- [ ] 2.1.4 Refactor `nifti_tool_get_series_info()`: Remove session parameter, use `create_many()`
- [ ] 2.1.5 Refactor `dicom_tool_get_series_info()`: Use batch operations

### 2.2 Eliminate Raw SQL text()
- [ ] 2.2.1 Replace `_get_studies_ready_for_transfer()` raw SQL with Repository method
- [ ] 2.2.2 Replace `query_studies_pending_completion()` raw SQL with Repository method
- [ ] 2.2.3 Replace `identify_completed_studies()` raw SQL with Repository method
- [ ] 2.2.4 Replace `get_stydy_series_ope_no_status()` with Repository `list_and_count()`
- [ ] 2.2.5 Replace `get_stydy_ope_no_status()` with Repository `list_and_count()`

### 2.3 Unify Session Management
- [x] 2.3.1 Remove `session: AsyncSession` parameter from `nifti_tool_get_series_info()`
- [ ] 2.3.2 Ensure all methods use `self.session_manager.get_session()` consistently
- [ ] 2.3.3 Remove commented-out `self.repository.session` references

## 3. Code Quality Improvements

### 3.1 Fix Minor Issues
- [x] 3.1.1 Fix typo: `flage` → `flag` (4 occurrences)
- [x] 3.1.2 Remove no-effect statement at line 182
- [x] 3.1.3 Fix `logger.error(traceback.print_exc())` → proper exception logging

### 3.2 Error Handling Enhancement
- [ ] 3.2.1 Replace generic `except Exception` with specific SQLAlchemy exceptions
- [ ] 3.2.2 Add proper error context to log messages
- [ ] 3.2.3 Ensure all exceptions are re-raised after rollback

## 4. Testing and Validation

### 4.1 Contract Tests
- [ ] 4.1.1 Write contract test: `post_ope_no_task()` behavior equivalence
- [ ] 4.1.2 Write contract test: `query_studies_pending_completion()` result equivalence
- [ ] 4.1.3 Write contract test: `identify_completed_studies()` result equivalence
- [ ] 4.1.4 Write contract test: pagination methods result equivalence

### 4.2 Performance Validation
- [ ] 4.2.1 Benchmark `post_ope_no_task()` with 100 records: before vs after
- [ ] 4.2.2 Benchmark `add_study_new()` with batch insert
- [ ] 4.2.3 Document performance improvements in PR description

### 4.3 Code Quality Checks
- [x] 4.3.1 Run `uvx ty check backend/app/sync/` (pre-existing type errors unrelated to refactoring)
- [x] 4.3.2 Run `uvx ruff check backend/app/sync/ --fix`
- [x] 4.3.3 Run `uvx ruff format backend/app/sync/`

## 5. Documentation and Cleanup

### 5.1 Documentation
- [ ] 5.1.1 Update docstrings for refactored methods
- [ ] 5.1.2 Add Repository pattern usage examples to CLAUDE.md (if applicable)
- [ ] 5.1.3 Document new Repository methods in code comments

### 5.2 Sync V2 Alignment
- [ ] 5.2.1 Apply same refactoring pattern to `backend/app/services/sync_v2.py`
- [ ] 5.2.2 Ensure V1 and V2 services use identical Repository methods
- [ ] 5.2.3 Consider deprecation strategy for V1 (future work)

## Dependencies

- **Depends on**: `refactor-to-pure-functions` (configuration injection pattern)
- **Parallel with**: None
- **Blocks**: None

## Verification Checklist

Before marking complete:
- [ ] All existing tests pass
- [ ] No direct `session.add()/commit()/refresh()` in Service layer
- [ ] No `text("SELECT...")` in Service layer (all in Repository)
- [ ] `ty check` passes with 0 errors (note: pre-existing type errors unrelated to this refactoring)
- [x] `ruff check` passes
- [x] `ruff format` produces no changes
- [ ] Contract tests verify behavior equivalence
