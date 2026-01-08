# Spec: Series-Level Inference

## ADDED Requirements

### Requirement: Batch Inference Request Processing
The system SHALL accept batch inference requests containing multiple studies, each specifying series UIDs and model identifiers for targeted inference execution.

#### Scenario: Single study with multiple series
- **WHEN** user submits request with study_uid="study1", series_uids=["series1", "series2"], model_name="cmb_model"
- **THEN** system validates all series are SERIES_CONVERSION_COMPLETE
- **AND** queues inference task for specified series only
- **AND** returns inference_id with "queued" status

#### Scenario: Batch of multiple studies
- **WHEN** user submits batch with 5 study requests, each with different series and models
- **THEN** system processes all requests in parallel
- **AND** returns batch_id with array of 5 inference responses
- **AND** each response contains unique inference_id

#### Scenario: Empty batch request
- **WHEN** user submits request with empty "requests" array
- **THEN** system returns 400 Bad Request error
- **AND** error message states "requests array cannot be empty"

### Requirement: Series Validation
The system SHALL validate that all specified series have completed DICOM-to-NIFTI conversion before queueing inference tasks.

#### Scenario: All series ready for inference
- **WHEN** all requested series have status SERIES_CONVERSION_COMPLETE (200.195)
- **THEN** validation returns accepted=[all series], rejected=[]
- **AND** system proceeds to queue inference task

#### Scenario: Some series not ready
- **WHEN** series1, series2 are SERIES_CONVERSION_COMPLETE but series3 is SERIES_CONVERTING (200.155)
- **THEN** validation returns accepted=["series1", "series2"], rejected=[{"series_uid":"series3", "reason":"SERIES_CONVERTING", "current_status":"200.155"}]
- **AND** system queues task for series1 and series2 only
- **AND** response status is "partial"

#### Scenario: All series not ready
- **WHEN** all requested series are still SERIES_CONVERTING
- **THEN** validation returns accepted=[], rejected=[all series with reasons]
- **AND** system does NOT queue any task
- **AND** response status is "rejected"

#### Scenario: Series does not exist
- **WHEN** requested series_uid does not exist in database
- **THEN** validation returns rejected with reason "SERIES_NOT_FOUND"
- **AND** current_status is null

### Requirement: Model Selection
The system SHALL support flexible model identification through either model_name + model_version or model_id (UUID).

#### Scenario: Model selection by name and version
- **WHEN** request specifies model_name="aneurysm_model", model_version="v1.2.0"
- **THEN** system uses model_name + model_version for task parameters
- **AND** cache key includes both name and version

#### Scenario: Model selection by UUID
- **WHEN** request specifies model_id="48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6"
- **THEN** system uses model_id for task parameters
- **AND** cache key uses model_id
- **AND** model_name and model_version are ignored if present

#### Scenario: Missing model identifier
- **WHEN** request has neither (model_name + model_version) nor model_id
- **THEN** system returns 400 Bad Request error
- **AND** error message states "Either model_id or (model_name + model_version) required"

### Requirement: Inference Caching
The system SHALL cache inference results to prevent duplicate processing of identical series + model combinations within 24-hour TTL.

#### Scenario: Cache miss - first request
- **WHEN** user submits request for study1, [series1, series2], model1
- **THEN** cache lookup returns None
- **AND** system queues new inference task
- **AND** cache is updated with inference_id and timestamp
- **AND** response cache_status is "new"

#### Scenario: Cache hit - duplicate request
- **WHEN** user submits identical request within 24 hours
- **THEN** cache lookup returns existing inference_id
- **AND** system does NOT queue new task
- **AND** response cache_status is "cached"
- **AND** response includes inference_id from cache

#### Scenario: Cache expiration
- **WHEN** cached entry is older than 24 hours
- **THEN** cache lookup returns None
- **AND** system queues new inference task
- **AND** cache is updated with new inference_id

#### Scenario: Cache invalidation on completion
- **WHEN** inference completes (success or failure)
- **THEN** system updates cache entry with completion status
- **AND** cache entry remains for query but marked as complete
- **AND** new requests with same params create new inference tasks

#### Scenario: Cache key collision with different series order
- **WHEN** request A has series_uids=["series1", "series2"] and request B has series_uids=["series2", "series1"]
- **THEN** both requests use same cache key (sorted series UIDs)
- **AND** request B returns cached result from request A

### Requirement: Task Queue Management
The system SHALL dispatch series inference tasks to a dedicated queue that shares GPU resources with study-level inference queue through distributed frequency control.

#### Scenario: Queue task for ready series
- **WHEN** series validation passes for at least one series
- **THEN** system creates SERIES_INFERENCE_READY event (300.055)
- **AND** builds task params with dual deployment paths
- **AND** pushes task to series_inference_queue
- **AND** creates SERIES_INFERENCE_QUEUED event (300.105) with task_id
- **AND** returns inference_id to caller

#### Scenario: GPU resource sharing
- **WHEN** both task_pipeline_inference_queue and series_inference_queue have tasks
- **THEN** funboost distributed frequency control ensures qps=1 across both queues
- **AND** only one inference task (study or series level) runs at a time on GPU
- **AND** tasks are processed in fair order across both queues

#### Scenario: Queue position tracking
- **WHEN** task is queued
- **THEN** response includes queue_status with position and estimated_wait_seconds
- **AND** position reflects total tasks across both queues
- **AND** estimated_wait is based on average task duration

#### Scenario: Task parameter injection
- **WHEN** task is created for series inference
- **THEN** task params include: series_uid, study_uid, nifti_series_path, model parameters
- **AND** task params include dual deployment paths: path_process, path_json, path_log
- **AND** task params include upload_data_api_url for callback
- **AND** task params include inference_id for tracking

### Requirement: Status Tracking
The system SHALL track series inference lifecycle through DCOP event status codes.

#### Scenario: Status progression - success
- **WHEN** inference task executes successfully
- **THEN** status progresses: SERIES_INFERENCE_READY (300.055) → SERIES_INFERENCE_QUEUED (300.105) → SERIES_INFERENCE_RUNNING (300.155) → SERIES_INFERENCE_COMPLETE (300.205)
- **AND** each status transition creates new DCOP event
- **AND** events include inference_id for tracking

#### Scenario: Status progression - failure
- **WHEN** inference task fails
- **THEN** status progresses: SERIES_INFERENCE_READY → SERIES_INFERENCE_QUEUED → SERIES_INFERENCE_RUNNING → SERIES_INFERENCE_FAILED (300.195)
- **AND** SERIES_INFERENCE_FAILED event includes error details in result_data

#### Scenario: Status query by inference_id
- **WHEN** caller queries GET /inference/series/status/{inference_id}
- **THEN** system returns latest DCOP event for that inference_id
- **AND** response includes status code, timestamp, params_data, result_data

#### Scenario: Batch status aggregation
- **WHEN** caller queries GET /inference/series/batch/{batch_id}
- **THEN** system returns array of status for all inferences in batch
- **AND** includes counts: total, queued, running, completed, failed

### Requirement: Inference Result Callback
The system SHALL receive inference results from GPU workers via callback endpoint matching radax integration requirements.

#### Scenario: Successful inference callback
- **WHEN** GPU worker POSTs to /inference/series/complete with result="success"
- **THEN** system creates SERIES_INFERENCE_COMPLETE event
- **AND** stores resultData (CMB prediction.json structure) in event
- **AND** updates cache entry with completion status
- **AND** returns 200 OK response

#### Scenario: Failed inference callback
- **WHEN** GPU worker POSTs with result="failed"
- **THEN** system creates SERIES_INFERENCE_FAILED event
- **AND** stores error information in result_data
- **AND** returns 200 OK response

#### Scenario: Callback data validation
- **WHEN** callback payload is received
- **THEN** system validates required fields: studyInstanceUid, modelName, inferenceId, result
- **AND** validates result is either "success" or "failed"
- **AND** returns 400 Bad Request if validation fails

#### Scenario: Callback result structure for CMB model
- **WHEN** callback includes CMB inference results
- **THEN** resultData includes: inference_id, inference_timestamp, input_study_instance_uid, input_series_instance_uid, model_id, patient_id, detections array
- **AND** each detection includes: annotated_series_instance_uid, series_instance_uid, sop_instance_uid, label, type, location, diameter, main_seg_slice, probability, pitch_angle, yaw_angle, mask_index, sub_location

### Requirement: API Response Format
The system SHALL return structured responses with inference identifiers, validation results, and queue status information.

#### Scenario: Successful batch request response
- **WHEN** batch request is processed
- **THEN** response includes batch_id (UUID)
- **AND** results array with one entry per request
- **AND** each result includes: study_uid, inference_id, model_name, model_version, queue_status, series_validation, cache_status, status

#### Scenario: Queue status information
- **WHEN** response includes queue_status
- **THEN** queue_status contains: position (integer), estimated_wait_seconds (integer)
- **AND** position reflects current queue depth across both queues
- **AND** estimated_wait_seconds = position * average_task_duration

#### Scenario: Series validation details
- **WHEN** response includes series_validation
- **THEN** series_validation contains: accepted (array of series UIDs), rejected (array of objects)
- **AND** each rejected entry includes: series_uid, reason (string), current_status (status code)

#### Scenario: Error response format
- **WHEN** request processing fails
- **THEN** response status is 4xx or 5xx
- **AND** response body includes: detail (error message), batch_id (if batch created), failed_requests (array of failures)

### Requirement: Cache Management API
The system SHALL provide endpoints for cache inspection and management.

#### Scenario: List cache entries
- **WHEN** caller sends GET /inference/cache with study_uid filter
- **THEN** system returns all cache entries for that study
- **AND** each entry includes: cache_key, inference_id, study_uid, series_uids, model_name, model_version, timestamp, status

#### Scenario: Clear cache by study
- **WHEN** caller sends DELETE /inference/cache?study_uid=study1
- **THEN** system deletes all cache entries for study1
- **AND** returns count of deleted entries

#### Scenario: Clear all cache
- **WHEN** caller sends DELETE /inference/cache?all=true
- **THEN** system deletes all inference cache entries
- **AND** returns count of deleted entries
- **AND** requires admin authentication

### Requirement: Partial Processing
The system SHALL support partial processing where ready series are queued immediately while not-ready series are recorded as rejected.

#### Scenario: Mixed readiness in single request
- **WHEN** request has 4 series: 2 ready, 2 not ready
- **THEN** system queues task for 2 ready series
- **AND** returns status "partial"
- **AND** series_validation shows accepted=[2 ready], rejected=[2 not ready with reasons]
- **AND** inference_id is created for the 2 ready series

#### Scenario: All series ready
- **WHEN** all requested series are SERIES_CONVERSION_COMPLETE
- **THEN** system queues task for all series
- **AND** returns status "queued"
- **AND** series_validation shows accepted=[all], rejected=[]

#### Scenario: No series ready
- **WHEN** no requested series are ready
- **THEN** system does NOT queue any task
- **AND** returns status "rejected"
- **AND** series_validation shows accepted=[], rejected=[all with reasons]
- **AND** no inference_id is created

### Requirement: Dual Deployment Support
The system SHALL support dual deployment architecture where multiple backends share GPU workers through parameter injection.

#### Scenario: Path parameter injection
- **WHEN** backend creates inference task
- **THEN** task params include path_process, path_json, path_log from backend config
- **AND** task params include upload_data_api_url for callback routing
- **AND** GPU worker uses injected paths instead of environment variables

#### Scenario: Callback routing
- **WHEN** GPU worker completes inference
- **THEN** worker POSTs results to upload_data_api_url from task params
- **AND** callback reaches correct backend that created the task
- **AND** backend updates its DCOP events accordingly

#### Scenario: Fallback to environment variables
- **WHEN** task params do not include path_process
- **THEN** GPU worker falls back to PATH_PROCESS environment variable
- **AND** same fallback logic for path_json, path_log
- **AND** warning is logged about missing parameter injection
