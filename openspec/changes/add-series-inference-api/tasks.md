# Implementation Tasks: Add Series-Level Inference API

## 1. Schema Definition & Status Codes
- [x] 1.1 Extend DCOPStatus enum with series inference codes in `backend/app/sync/schemas.py`
- [x] 1.2 Create `backend/app/inference/schemas.py` with all request/response models
- [x] 1.3 Define SeriesInferenceRequest, BatchInferenceRequest, InferenceResponse
- [x] 1.4 Define SeriesValidationResult, QueueStatus, CacheStatus models
- [x] 1.5 Add model selection schemas (ModelIdentifier with name+version or id)
- [x] 1.6 Validate schema definitions with Pydantic type checking

## 2. Database Model Setup
- [x] 2.1 Create `backend/app/inference/model.py` importing DCOP models
- [x] 2.2 Verify DCOPConfModel supports new status codes
- [x] 2.3 Create database conf entries for SERIES_INFERENCE_* statuses
- [ ] 2.4 Test event creation with new status codes
- [ ] 2.5 Verify event queries work with inference_id tracking

## 3. Redis Cache Implementation
- [x] 3.1 Add Redis dependency to `backend/app/inference/deps.py`
- [x] 3.2 Implement cache key generation function (hash series UIDs + model)
- [x] 3.3 Implement cache_get(key) → Optional[CacheEntry]
- [x] 3.4 Implement cache_set(key, entry, ttl=86400)
- [x] 3.5 Implement cache_delete(key) and cache_clear(pattern)
- [x] 3.6 Add cache statistics tracking (total entries, memory bytes, timestamps)

## 4. Service Layer - Validation Methods
- [x] 4.1 Create `backend/app/inference/service.py` with DCOPEventInferenceService class
- [x] 4.2 Implement validate_series_ready(study_uid, series_uids) → ValidationResult
- [x] 4.3 Query database for series conversion status (SERIES_CONVERSION_COMPLETE)
- [x] 4.4 Return accepted/rejected series lists with reasons
- [x] 4.5 Handle partial processing logic (some ready, some not)
- [x] 4.6 Add logging for validation decisions

## 5. Service Layer - Cache Management
- [x] 5.1 Implement check_inference_cache(study_uid, series_uids, model_id)
- [x] 5.2 Compute cache key from sorted series UIDs + model identifier
- [ ] 5.3 Check Redis for existing inference
- [ ] 5.4 Return cached inference_id or None
- [ ] 5.5 Add cache hit/miss metrics logging
- [ ] 5.6 Implement cache invalidation on inference completion

## 6. Service Layer - Task Queuing
- [x] 6.1 Implement queue_series_inference_task(params) → inference_id
- [x] 6.2 Create SERIES_INFERENCE_READY event with params
- [x] 6.3 Build task params with dual deployment paths
- [x] 6.4 Push task to task_pipeline_inference.push(params) (統一入口)
- [x] 6.5 Create SERIES_INFERENCE_QUEUED event with task_id
- [ ] 6.6 Update Redis cache with task_id and timestamp
- [ ] 6.7 Send events via _send_events() callback

## 7. Service Layer - Main Orchestration
- [x] 7.1 Implement process_batch_inference(data: BatchInferenceRequest)
- [x] 7.2 Iterate through batch requests
- [x] 7.3 For each request: validate → check cache → queue or return cached
- [x] 7.4 Build batch response with all inference_ids and validation results
- [x] 7.5 Handle errors gracefully with detailed error messages
- [ ] 7.6 Add transaction handling for database operations
- [x] 7.7 Return complete BatchInferenceResponse

## 8. Service Layer - Status & Completion
- [ ] 8.1 Implement get_inference_status(inference_id) → StatusResponse
- [ ] 8.2 Query DCOP events for inference_id across all statuses
- [ ] 8.3 Return current status, progress, and results if complete
- [ ] 8.4 Implement get_batch_status(batch_id) → BatchStatusResponse
- [x] 8.5 Implement handle_inference_complete(callback_data) - (透過 task_pipeline_inference 完成事件發送)
- [x] 8.6 Update DCOP events with completion status (success/failed) - (task_pipeline.py 中實作)
- [x] 8.7 Store result_data in DCOP event - (task_pipeline.py 中實作)
- [ ] 8.8 Clear or update cache entry on completion

## 9. Router Endpoints
- [x] 9.1 Create `backend/app/inference/routers.py` with APIRouter
- [x] 9.2 Implement POST /inference/series → queue_series_inference()
- [x] 9.3 Implement GET /inference/series/status/{inference_id} → get_inference_status()
- [ ] 9.4 Implement GET /inference/series/batch/{batch_id}
- [x] 9.5 Implement POST /inference/series/complete (callback from GPU worker) → inference_complete_callback()
- [x] 9.6 Implement GET /inference/cache (list cache entries) → list_cache()
- [x] 9.7 Implement DELETE /inference/cache (clear cache) → clear_cache()
- [x] 9.8 Add proper error handling with HTTPException
- [x] 9.9 Add request validation with Pydantic
- [ ] 9.10 Add OpenAPI documentation with examples

## 10. Dependencies & URLs
- [x] 10.1 Create `backend/app/inference/deps.py` with dependency injection functions
- [x] 10.2 Add get_inference_service() dependency
- [ ] 10.3 Add get_redis_client() dependency if needed
- [x] 10.4 Create `backend/app/inference/urls.py` with path definitions
- [x] 10.5 Define all endpoint paths as constants
- [x] 10.6 Create `backend/app/inference/__init__.py` with exports

## 11. Task Queue Implementation (統一入口模式 - 已整合至 task_pipeline.py)
- [x] 11.1 ~~Create `code_ai/task/task_series_inference.py`~~ → 已整合至 task_pipeline.py 的 _task_series_pipeline_inference()
- [x] 11.2 使用統一入口 task_pipeline_inference (qps=1 提供自然 GPU 互斥)
- [x] 11.3 Implement _task_series_pipeline_inference(func_params) function
- [x] 11.4 Extract paths from params (dual deployment support)
- [x] 11.5 Locate NIFTI file for specified series (支援轉換模式與直接模式)
- [x] 11.6 Build inference command for specified model (_build_series_inference_cmd)
- [x] 11.7 Execute inference subprocess with timeout handling (600s timeout)
- [x] 11.8 Parse inference results (prediction.json format)
- [x] 11.9 Post results via DCOP events (SERIES_INFERENCE_COMPLETE/FAILED)
- [x] 11.10 Handle errors and post failure status
- [x] 11.11 Add comprehensive logging throughout task execution

## 12. Task Queue Integration (統一入口模式 - 單一 Queue)
- [x] 12.1 ~~Update `code_ai/task/__init__.py`~~ → 無需額外 export，使用統一入口 task_pipeline_inference
- [x] 12.2 ~~Verify task is registered~~ → 統一入口已自動註冊，series_uids 判斷行為
- [x] 12.3 ~~Update funboost_cli_user.py~~ → 無需修改，單一 queue 已消費兩種模式
- [x] 12.4 ~~BoostersManager.consume_queues()~~ → 單一 queue (qps=1) 提供自然 GPU 互斥
- [ ] 12.5 Test queue consumer startup
- [x] 12.6 ~~Verify distributed frequency control~~ → 由 qps=1 自然提供，無需額外協調

## 13. Main Application Integration
- [x] 13.1 Update `backend/app/routers.py` to import inference router
- [x] 13.2 Add router.include_router() with inference tag (在 routers.py 中)
- [x] 13.3 Add "inference" tag to OpenAPI docs
- [x] 13.4 Verify router registration in FastAPI app
- [ ] 13.5 Test API documentation at /docs endpoint

## 14. Unit Tests - Schemas
- [ ] 14.1 Test SeriesInferenceRequest validation (valid cases)
- [ ] 14.2 Test SeriesInferenceRequest validation (invalid cases)
- [ ] 14.3 Test BatchInferenceRequest with multiple requests
- [ ] 14.4 Test model selection logic (name+version vs model_id)
- [ ] 14.5 Test response schema serialization

## 15. Unit Tests - Service Layer
- [ ] 15.1 Test validate_series_ready() with all ready series
- [ ] 15.2 Test validate_series_ready() with mixed ready/not-ready
- [ ] 15.3 Test validate_series_ready() with all not-ready series
- [ ] 15.4 Test check_inference_cache() cache hit scenario
- [ ] 15.5 Test check_inference_cache() cache miss scenario
- [ ] 15.6 Test queue_series_inference_task() event creation
- [ ] 15.7 Test process_batch_inference() with valid batch
- [ ] 15.8 Test process_batch_inference() with partial processing
- [ ] 15.9 Test handle_inference_complete() success case
- [ ] 15.10 Test handle_inference_complete() failure case

## 16. Integration Tests - API Endpoints
- [ ] 16.1 Test POST /inference/series with single study request
- [ ] 16.2 Test POST /inference/series with batch of 5 studies
- [ ] 16.3 Test POST /inference/series with cache hit (duplicate request)
- [ ] 16.4 Test POST /inference/series with partial processing scenario
- [ ] 16.5 Test GET /inference/series/status/{id} for queued inference
- [ ] 16.6 Test GET /inference/series/status/{id} for completed inference
- [ ] 16.7 Test GET /inference/series/batch/{id} for batch status
- [ ] 16.8 Test POST /inference/series/complete callback
- [ ] 16.9 Test GET /inference/cache listing
- [ ] 16.10 Test DELETE /inference/cache clearing

## 17. Integration Tests - Queue Consumer
- [ ] 17.1 Test series_inference queue consumes tasks
- [ ] 17.2 Test task execution with valid NIFTI series
- [ ] 17.3 Test task error handling (missing NIFTI file)
- [ ] 17.4 Test callback posting on success
- [ ] 17.5 Test callback posting on failure
- [ ] 17.6 Test dual deployment path injection
- [ ] 17.7 Test concurrent queue consumption with study-level queue

## 18. Integration Tests - Cache Behavior
- [ ] 18.1 Test cache prevents duplicate requests within TTL
- [ ] 18.2 Test cache expiration after TTL (24 hours)
- [ ] 18.3 Test cache invalidation on completion
- [ ] 18.4 Test cache key collision scenarios
- [ ] 18.5 Test cache clear operation
- [ ] 18.6 Test cache performance under load

## 19. End-to-End Testing
- [ ] 19.1 Full workflow: API request → validation → queuing → task execution → callback
- [ ] 19.2 Test with actual NIFTI files and model inference
- [ ] 19.3 Test batch of 10 studies with different models
- [ ] 19.4 Test partial processing: mix of ready and not-ready series
- [ ] 19.5 Test cache behavior across multiple requests
- [ ] 19.6 Test GPU queue fairness (series-level vs study-level)
- [ ] 19.7 Test error scenarios and rollback
- [ ] 19.8 Test API performance under concurrent requests
- [ ] 19.9 Verify DCOP event tracking completeness
- [ ] 19.10 Verify callback matches radax requirements

## 20. Documentation & Deployment
- [ ] 20.1 Update API documentation with new endpoints
- [ ] 20.2 Add usage examples for batch inference requests
- [ ] 20.3 Document cache configuration and TTL settings
- [ ] 20.4 Document queue consumer startup procedures
- [ ] 20.5 Create deployment checklist
- [ ] 20.6 Update architecture diagrams
- [ ] 20.7 Document model selection strategies
- [ ] 20.8 Add troubleshooting guide
- [ ] 20.9 Create OpenSpec archive documentation
- [ ] 20.10 Validate all tasks completed and tests passing
