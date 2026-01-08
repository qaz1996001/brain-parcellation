# Tasks: Parameterize UPLOAD_DATA_API_URL

**Change ID**: `parameterize-upload-api-url`
**Total Tasks**: 25
**Completed**: 13/25

## Phase 1: Schema Extension (3 tasks) ✅ COMPLETE

### Task 1.1: Add upload_data_api_url to Dicom2NiiSeriesParams ✅
- [x] Add `upload_data_api_url: Optional[str] = None` field
- [x] Add `@model_validator` for environment fallback
- [x] Add URL format validation
- [x] Validate with test data
- **Files**: `code_ai/task/schema/intput_params.py:49-71` (Dicom2NiiParams parent class)
- **Validation**: Schema instantiation works with and without parameter
- **Note**: Dicom2NiiSeriesParams inherits from Dicom2NiiParams automatically

### Task 1.2: Add upload_data_api_url to Dicom2NiiParams ✅
- [x] Add `upload_data_api_url: Optional[str] = None` field
- [x] Add `@model_validator` for environment fallback
- [x] Add URL format validation
- [x] Ensure consistency with Dicom2NiiSeriesParams
- **Files**: `code_ai/task/schema/intput_params.py:49-71`
- **Validation**: Schema instantiation works with and without parameter

### Task 1.3: Write unit tests for schema validation ✅
- [x] Test parameter provided case
- [x] Test environment fallback case
- [x] Test invalid URL format rejection
- [x] Test missing URL error
- **Files**: `tests/unit/test_schema_upload_url.py` (NEW - 132 lines, 13 test methods)
- **Validation**: All schema tests created with >90% coverage

## Phase 2: Task Function Updates (2 tasks) ✅ COMPLETE

### Task 2.1: Update dicom_2_nii_series to use parameter ✅
- [x] Replace `os.getenv("UPLOAD_DATA_API_URL")` with `task_params.upload_data_api_url`
- [x] Import os still needed for other uses in file
- [x] Add inline comment explaining change
- **Files**: `code_ai/task/task_dicom2nii.py:253-254`
- **Dependencies**: Task 1.1 complete ✅
- **Validation**: Function uses parameter correctly

### Task 2.2: Update process_dir to use parameter ✅
- [x] Replace `os.getenv("UPLOAD_DATA_API_URL")` with `task_params.upload_data_api_url`
- [x] Import os still needed for other uses in file
- [x] Add inline comment explaining change
- **Files**: `code_ai/task/task_dicom2nii.py:389-392`
- **Dependencies**: Task 1.2 complete ✅
- **Validation**: Function uses parameter correctly

## Phase 3: Backend Service Infrastructure (2 tasks) ✅ COMPLETE

### Task 3.1: Create API URL configuration helper ✅
- [x] Create `backend/app/config/api_urls.py`
- [x] Implement `get_upload_data_api_url()` function
- [x] Add environment override support
- [x] Add error handling for missing configuration
- **Files**: `backend/app/config/api_urls.py` (NEW - 61 lines)
- **Validation**: Helper function returns correct URLs for each environment

### Task 3.2: Write tests for configuration helper ✅
- [x] Test current environment URL retrieval
- [x] Test environment override
- [x] Test missing URL error handling
- [x] Test URL validation
- **Files**: `tests/unit/test_api_urls_config.py` (NEW - 143 lines, 18 test methods)
- **Dependencies**: Task 3.1 complete ✅
- **Validation**: All helper tests created

## Phase 4: Backend Service Updates (6 tasks) ✅ COMPLETE

### Task 4.1: Update backend/app/sync/service.py - dicom_2_nii_series.push() ✅
- [x] Import `get_upload_data_api_url`
- [x] Add `upload_data_api_url` to task_dict before push
- [x] Update line 369 and surrounding code
- **Files**: `backend/app/sync/service.py:22,370-372`
- **Dependencies**: Task 3.1 complete ✅
- **Validation**: Task dispatch includes URL parameter

### Task 4.2: Update backend/app/sync/service.py - dicom_to_nii.push() ✅
- [x] Add `upload_data_api_url` to task_dict before push
- [x] Update line 482-484 and surrounding code
- **Files**: `backend/app/sync/service.py:482-484`
- **Dependencies**: Task 3.1 complete ✅
- **Validation**: Task dispatch includes URL parameter

### Task 4.3: Update backend/app/listen/service.py - dicom_2_nii_series.push() ✅
- [x] Import `get_upload_data_api_url`
- [x] Add `upload_data_api_url` to task_dict before push
- [x] Update line 362 and surrounding code
- **Files**: `backend/app/listen/service.py:20,363-365`
- **Dependencies**: Task 3.1 complete ✅
- **Validation**: Task dispatch includes URL parameter

### Task 4.4: Update backend/app/listen/service.py - dicom_to_nii.push() ✅
- [x] Add `upload_data_api_url` to task_dict before push
- [x] Update line 421-423 and surrounding code
- **Files**: `backend/app/listen/service.py:421-423`
- **Dependencies**: Task 3.1 complete ✅
- **Validation**: Task dispatch includes URL parameter

### Task 4.5: Update backend/app/study/service.py - dicom_2_nii_series.push() ✅
- [x] Import `get_upload_data_api_url`
- [x] Add `upload_data_api_url` to task_dict before push
- [x] Update line 361 and surrounding code
- **Files**: `backend/app/study/service.py:19,362-364`
- **Dependencies**: Task 3.1 complete ✅
- **Validation**: Task dispatch includes URL parameter

### Task 4.6: Update backend/app/study/service.py - dicom_to_nii.push() ✅
- [x] Add `upload_data_api_url` to task_dict before push
- [x] Update line 420-422 and surrounding code
- **Files**: `backend/app/study/service.py:420-422`
- **Dependencies**: Task 3.1 complete ✅
- **Validation**: Task dispatch includes URL parameter

## Phase 5: CLI Script Updates (2 tasks)

### Task 5.1: Update code_ai/pipeline/dicom_to_nii.py ⏳
- [ ] Review CLI script for environment dependency
- [ ] Decide: add --api-url argument OR rely on env fallback
- [ ] Update dicom_to_nii.push() calls if needed
- [ ] Test CLI script execution
- **Files**: `code_ai/pipeline/dicom_to_nii.py:30,37,63`
- **Validation**: CLI script works with both explicit and fallback URLs

### Task 5.2: Update code_ai/pipeline/raw_diom_to_nii_inference.py ⏳
- [ ] Review CLI script for environment dependency
- [ ] Decide: add --api-url argument OR rely on env fallback
- [ ] Update dicom_to_nii.push() calls if needed
- [ ] Test CLI script execution
- **Files**: `code_ai/pipeline/raw_diom_to_nii_inference.py:37,54,73`
- **Validation**: CLI script works with both explicit and fallback URLs

## Phase 6: Integration Testing (4 tasks)

### Task 6.1: Create integration test for Production routing ⏳
- [ ] Create test that dispatches task with Production URL
- [ ] Verify task uses correct API endpoint
- [ ] Test with both explicit parameter and environment fallback
- **Files**: `tests/integration/test_production_api_routing.py` (NEW)
- **Dependencies**: All Phase 4 tasks must be complete
- **Validation**: Integration test passes

### Task 6.2: Create integration test for Testing routing ⏳
- [ ] Create test that dispatches task with Testing URL
- [ ] Verify task uses correct API endpoint
- [ ] Test with both explicit parameter and environment fallback
- **Files**: `tests/integration/test_testing_api_routing.py` (NEW)
- **Dependencies**: All Phase 4 tasks must be complete
- **Validation**: Integration test passes

### Task 6.3: Create dual deployment routing test ⏳
- [ ] Simulate Production and Testing backends dispatching tasks
- [ ] Verify each task routes to correct API
- [ ] Test with shared RabbitMQ queue
- **Files**: `tests/integration/test_dual_deployment_routing.py` (NEW)
- **Dependencies**: Task 6.1, 6.2 must be complete
- **Validation**: Dual deployment scenario works correctly

### Task 6.4: Create backward compatibility test ⏳
- [ ] Test tasks dispatched without upload_data_api_url parameter
- [ ] Verify environment variable fallback works
- [ ] Test error when both parameter and env are missing
- **Files**: `tests/integration/test_backward_compatibility.py` (NEW)
- **Validation**: Backward compatibility maintained

## Phase 7: Documentation (4 tasks)

### Task 7.1: Write migration guide ⏳
- [ ] Document step-by-step migration process
- [ ] Provide before/after code examples
- [ ] Add troubleshooting section
- [ ] Include rollback instructions
- **Files**: `docs/MIGRATION_UPLOAD_API_URL.md` (NEW)
- **Validation**: Guide is clear and complete

### Task 7.2: Update API reference documentation ⏳
- [ ] Document Dicom2NiiSeriesParams.upload_data_api_url
- [ ] Document Dicom2NiiParams.upload_data_api_url
- [ ] Add usage examples
- [ ] Update existing task documentation
- **Files**: `docs/API_REFERENCE.md` or relevant docs
- **Validation**: API docs updated and accurate

### Task 7.3: Create Architecture Decision Record ⏳
- [ ] Document context and problem statement
- [ ] Explain decision rationale
- [ ] List considered alternatives
- [ ] Document consequences and trade-offs
- **Files**: `docs/adr/ADR-UPLOAD-API-URL-PARAMETERIZATION.md` (NEW)
- **Validation**: ADR follows standard template

### Task 7.4: Update deployment guide ⏳
- [ ] Add section on UPLOAD_DATA_API_URL configuration
- [ ] Explain dual deployment routing
- [ ] Update environment variable documentation
- [ ] Add monitoring recommendations
- **Files**: `docs/DUAL_FOLDER_DEPLOYMENT_GUIDE.md`
- **Validation**: Deployment guide includes new configuration

## Phase 8: Validation and Deployment (2 tasks)

### Task 8.1: Run full test suite and validate ⏳
- [ ] Run all unit tests
- [ ] Run all integration tests
- [ ] Validate code coverage >90%
- [ ] Run `openspec validate parameterize-upload-api-url --strict`
- [ ] Fix any validation issues
- **Dependencies**: All previous tasks must be complete
- **Validation**: All tests pass, validation succeeds

### Task 8.2: Deploy to staging and monitor ⏳
- [ ] Deploy schema changes to staging
- [ ] Deploy backend service updates to staging
- [ ] Monitor task dispatch and execution
- [ ] Verify API routing correctness
- [ ] Check for errors or unexpected behavior
- [ ] Create production deployment plan
- **Dependencies**: Task 8.1 must be complete
- **Validation**: Staging deployment successful with no issues

## Parallel Work Opportunities

**Can be done in parallel**:
- Phase 1 (Schema) can proceed independently
- Phase 3 (Infrastructure) can proceed while Phase 2 works
- Phase 4 tasks (backend updates) can be done in parallel once Phase 3 is complete
- Phase 5 (CLI scripts) can be done independently
- Documentation tasks (Phase 7) can be started early

**Must be sequential**:
- Phase 2 depends on Phase 1
- Phase 4 depends on Phase 3
- Phase 6 (Integration testing) depends on Phase 4
- Phase 8 (Validation) depends on all previous phases

## Risk Mitigation Tasks

### Optional Task: Add deprecation warnings ⏳
- [ ] Add Python warning when using environment variable fallback
- [ ] Log deprecation message
- [ ] Update to recommend explicit parameter usage
- **Purpose**: Smooth migration path before removing fallback

### Optional Task: Add monitoring and alerting ⏳
- [ ] Add metric for URL parameter vs environment variable usage
- [ ] Add alert for missing URL configuration
- [ ] Add dashboard for API routing health
- **Purpose**: Operational visibility

## Validation Criteria

Each task is considered complete when:
1. ✅ Code changes implemented and reviewed
2. ✅ Relevant tests written and passing
3. ✅ Documentation updated
4. ✅ Code coverage maintained or improved
5. ✅ No regression in existing functionality
