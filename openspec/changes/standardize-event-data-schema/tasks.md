# Implementation Tasks

## 1. Schema Foundation

- [ ] 1.1 Create `backend/app/sync/schemas/params/` directory structure
- [ ] 1.2 Create `backend/app/sync/schemas/results/` directory structure
- [ ] 1.3 Implement `BaseParamsData` class in `params/base.py`
  - Common fields: timestamp, tool_id, ope_no
  - JSON encoders for Path → string conversion
  - Pydantic Config for from_attributes
- [ ] 1.4 Implement `BaseResultData` class in `results/base.py`
  - Common fields: timestamp, status, message
  - Status as Literal["success", "error", "warning"]
  - JSON encoders configuration
- [ ] 1.5 Add unit tests for base schemas
  - Test field validation
  - Test Path serialization
  - Test timestamp auto-generation

## 2. Transfer Operation Schemas

- [ ] 2.1 Implement `StudyTransferParams` in `params/transfer.py`
  - Fields: sub_dir, output_dicom_path, output_nifti_path
  - Validator: ensure absolute paths
  - Docstring with usage example
- [ ] 2.2 Implement `SeriesTransferParams` in `params/transfer.py`
  - Inherit appropriate fields
  - Add series-specific fields
- [ ] 2.3 Implement `StudyTransferResult` in `results/transfer.py`
  - Fields: study_id, series_count, total_files, total_size_mb
  - Dict field for output_paths mapping
  - Optional lists for skipped_series, errors
- [ ] 2.4 Implement `SeriesTransferResult` in `results/transfer.py`
  - Fields: series_uid, file_count, size_mb
- [ ] 2.5 Add unit tests for transfer schemas
  - Test valid data acceptance
  - Test validation errors for invalid data
  - Test path validation

## 3. Conversion Operation Schemas

- [ ] 3.1 Implement `StudyConversionParams` in `params/conversion.py`
  - Fields: study_uid, output_dicom_path, output_nifti_path
  - Validation for required fields
- [ ] 3.2 Implement `SeriesConversionParams` in `params/conversion.py`
  - Fields: study_uid, series_uid, output_dicom_path, output_nifti_path
  - Optional sub_dir field
- [ ] 3.3 Implement `StudyConversionResult` in `results/conversion.py`
  - Fields: study_id, series_count, conversion_summary
- [ ] 3.4 Implement `SeriesConversionResult` in `results/conversion.py`
  - Fields: study_uid, series_uid, nifti_file_path, json_file_path
  - Fields: series_description, modality, file_size_mb, conversion_duration_sec
  - Validation for numeric fields (size > 0, duration >= 0)
- [ ] 3.5 Add unit tests for conversion schemas
  - Test required vs optional fields
  - Test numeric field validation
  - Test path field serialization

## 4. Inference Operation Schemas

- [ ] 4.1 Implement `StudyInferenceParams` in `params/inference.py`
  - Fields: nifti_study_path, dicom_study_path, study_uid, study_id
  - Path validation
- [ ] 4.2 Implement `StudyInferenceResult` in `results/inference.py`
  - Fields: study_uid, study_id, inference_outputs
  - Fields: task_pipeline_id, duration_sec
- [ ] 4.3 Add unit tests for inference schemas

## 5. Schema Registry

- [ ] 5.1 Create `backend/app/sync/schemas/registry.py`
- [ ] 5.2 Implement `PARAMS_SCHEMA_REGISTRY` dict
  - Map DCOPStatus operation codes to params schema classes
  - Include all transfer, conversion, inference operations
- [ ] 5.3 Implement `RESULT_SCHEMA_REGISTRY` dict
  - Map DCOPStatus operation codes to result schema classes
- [ ] 5.4 Implement `get_params_schema(ope_no: str)` function
  - Return schema class for given operation
  - Fallback to BaseParamsData for unknown operations
  - Log warning for unknown operations
- [ ] 5.5 Implement `get_result_schema(ope_no: str)` function
  - Return schema class for given operation
  - Fallback to BaseResultData for unknown operations
- [ ] 5.6 Add unit tests for registry
  - Test schema lookup for all operation codes
  - Test fallback behavior for unknown codes
  - Test registry completeness (all operations covered)

## 6. Model Integration

- [ ] 6.1 Update `DCOPEventModel.create_event_ope_no` method in `backend/app/sync/model.py`
  - Import schema registry functions
  - Validate params_data using get_params_schema
  - Validate result_data using get_result_schema
  - Convert validated schemas to dict for JSON column
  - Handle ValidationError with clear messages
- [ ] 6.2 Add optional parameter `validate: bool = True` to support gradual rollout
- [ ] 6.3 Add unit tests for model validation integration
  - Test successful validation flow
  - Test validation error handling
  - Test validation bypass when validate=False

## 7. Service Layer Updates

- [ ] 7.1 Update `backend/app/sync/service.py` methods to use typed schemas
  - `post_ope_no_task`: validate DCOPEventRequest fields
  - `_get_studies_ready_for_transfer`: use StudyTransferResult schema
  - `_create_study_complete_events`: use StudyConversionResult schema
  - `_queue_inference_tasks`: use StudyInferenceParams schema
- [ ] 7.2 Update `backend/app/study/service.py` to use schemas
  - Event creation calls with typed params_data
- [ ] 7.3 Update `backend/app/listen/service.py` to use schemas
  - Event creation calls with typed params_data
- [ ] 7.4 Add integration tests for service layer
  - Test end-to-end event creation with validation
  - Test error handling for invalid data
  - Test backward compatibility with existing events

## 8. Task Module Updates

- [ ] 8.1 Update `code_ai/task/task_dicom2nii.py`
  - Use typed schema classes instead of raw dicts
  - Import relevant schemas from backend.app.sync.schemas
- [ ] 8.2 Update `code_ai/task/task_pipeline.py`
  - Use typed schema classes for inference parameters
- [ ] 8.3 Update `code_ai/task/schema/intput_params.py` if needed
  - Align with new schema structure
  - Consider deprecating duplicate schemas
- [ ] 8.4 Add integration tests for task modules
  - Test task parameter validation
  - Test task result validation

## 9. Query Helper Methods

- [ ] 9.1 Create `backend/app/sync/query_helpers.py`
  - Implement `extract_params_field(event, field_name)` helper
  - Implement `extract_result_field(event, field_name)` helper
  - Implement SQL JSON query builders for common patterns
- [ ] 9.2 Add utility functions for schema-aware queries
  - `get_all_output_dicom_paths(session, ope_no)`
  - `get_all_nifti_file_paths(session, study_uid)`
  - `get_conversion_metrics(session, study_uid)`
- [ ] 9.3 Add unit tests for query helpers
  - Test field extraction with valid schemas
  - Test field extraction with legacy data
  - Test SQL query generation

## 10. Documentation

- [ ] 10.1 Add docstrings to all schema classes
  - Class-level purpose documentation
  - Field-level descriptions using Field(description=...)
  - Usage examples in docstrings
- [ ] 10.2 Create schema reference documentation in `docs/schemas/`
  - Document each operation type's schemas
  - Provide usage examples
  - Include migration guide for developers
- [ ] 10.3 Update API documentation
  - Document schema validation in event creation endpoints
  - Document error response formats
- [ ] 10.4 Create developer migration guide
  - How to use new schemas
  - How to handle validation errors
  - Backward compatibility considerations

## 11. Testing & Validation

- [ ] 11.1 Add schema validation tests
  - Test all required fields for each schema
  - Test optional field handling
  - Test type coercion and validation
  - Test error messages for common mistakes
- [ ] 11.2 Add integration tests for end-to-end workflows
  - Test study transfer workflow with validation
  - Test series conversion workflow with validation
  - Test inference workflow with validation
- [ ] 11.3 Add backward compatibility tests
  - Test reading legacy events without validation errors
  - Test mixed environment (old + new events)
- [ ] 11.4 Performance testing
  - Measure validation overhead
  - Ensure <50ms additional latency
- [ ] 11.5 Add data migration validation script
  - Script to validate existing data against schemas
  - Report on data quality issues
  - No automatic migration, just analysis

## 12. Deployment

- [ ] 12.1 Create feature flag for schema validation
  - Environment variable: ENABLE_EVENT_SCHEMA_VALIDATION
  - Default: True for new deployments
- [ ] 12.2 Add monitoring for validation errors
  - Log validation failures
  - Metrics for validation error rates
  - Alerts for high error rates
- [ ] 12.3 Gradual rollout plan
  - Phase 1: Deploy with validation enabled, monitor
  - Phase 2: Fix any unexpected validation issues
  - Phase 3: Remove validation bypass code
- [ ] 12.4 Database query optimization
  - Add GIN indexes on commonly queried JSON fields
  - Test query performance with schemas
- [ ] 12.5 Update deployment documentation
  - Document new environment variables
  - Document monitoring and alerting
  - Document rollback procedure
