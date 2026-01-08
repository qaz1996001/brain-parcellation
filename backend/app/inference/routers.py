"""Series-level inference API routers.

Provides REST API endpoints for series-level inference:
- POST /inference/series - Queue series inference (single or batch)
- GET /inference/series/status/{inference_id} - Query inference status
- POST /inference/series/complete - Callback from GPU worker

Based on inference_params_design.md - data-driven design principles.
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.database import alchemy
from .service import DCOPEventInferenceService
from .schemas import (
    BatchInferenceRequest,
    InferenceResponse,
    BatchInferenceResponse,
    InferenceStatusResponse,
    InferenceCallbackRequest,
    CacheListResponse,
    CacheDeleteResponse,
    CacheStatistics,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
)


@router.post(
    "/series",
    status_code=status.HTTP_200_OK,
    response_model=InferenceResponse | BatchInferenceResponse,
    summary="Queue series-level inference",
    description="""
    Queue inference for specific series within a study.

    **Data-Driven Design** (Linus principle):
    - Presence of `series_uids` → series-level inference
    - No `series_uids` → would be study-level (use /sync endpoints instead)

    **Single Request**:
    ```json
    {
      "study_uid": "ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30",
      "series_uids": ["0c0a1444-9238e5ad-fbcd0251-335322e7-9af7b058"],
      "model_id": "uuid-model-123"
    }
    ```

    **Batch Request**:
    ```json
    {
      "requests": [
        {
          "study_uid": "study1",
          "series_uids": ["series1"],
          "model_id": "model1"
        },
        {
          "study_uid": "study2",
          "series_uids": ["series2", "series3"],
          "model_name": "aneurysm_model",
          "model_version": "v1.0"
        }
      ]
    }
    ```

    **Response**:
    - `status: "queued"` - All series accepted and queued
    - `status: "partial"` - Some series rejected (not ready)
    - `status: "rejected"` - All series rejected
    - `cache_status: "cached"` - Result from cache (duplicate request)

    **Validation**:
    - Series must have SERIES_CONVERSION_COMPLETE status
    - Rejected series will be listed with reasons
    - Partial processing is supported (some series ready, some not)
    """,
)
async def queue_series_inference(
    request: BatchInferenceRequest,
    inference_service: Annotated[
        DCOPEventInferenceService,
        Depends(alchemy.provide_service(DCOPEventInferenceService)),
    ],
) -> InferenceResponse | BatchInferenceResponse:
    """Queue series-level inference task.

    Args:
        request: Single or batch inference request
        inference_service: Inference service dependency

    Returns:
        Inference response with inference_id and validation results

    Raises:
        HTTPException: If request validation fails or queuing fails
    """
    try:
        # Linus: "Let the data structure do the talking"
        # Type discrimination happens automatically via Pydantic union type

        if isinstance(request, BatchInferenceRequest):
            logger.info(
                f"Processing batch inference request with {len(request.requests)} studies"
            )
            result = await inference_service.process_batch_inference(request)
        else:
            logger.info(
                f"Processing single inference request for study {request.study_uid}, "
                f"{len(request.series_uids)} series"
            )
            result = await inference_service.queue_series_inference(request)

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to queue inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue inference: {str(e)}",
        )


@router.get(
    "/series/status/{inference_id}",
    status_code=status.HTTP_200_OK,
    response_model=InferenceStatusResponse,
    summary="Query inference status",
    description="""
    Query the current status of an inference task by its ID.

    **Status Codes** (from DCOP events):
    - `300.055` - SERIES_INFERENCE_READY
    - `300.105` - SERIES_INFERENCE_QUEUED
    - `300.155` - SERIES_INFERENCE_RUNNING
    - `300.295` - SERIES_INFERENCE_COMPLETE

    **Example**:
    ```
    GET /inference/series/status/550e8400-e29b-41d4-a716-446655440000
    ```

    **Response**:
    ```json
    {
      "inference_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "300.155",
      "study_uid": "study-uid",
      "series_uids": ["series1", "series2"],
      "model_id": "model-uuid",
      "timestamp": "2024-01-15T10:30:00"
    }
    ```
    """,
)
async def get_inference_status(
    inference_id: UUID,
    inference_service: Annotated[
        DCOPEventInferenceService,
        Depends(alchemy.provide_service(DCOPEventInferenceService)),
    ],
) -> InferenceStatusResponse:
    """Query inference status by ID.

    Args:
        inference_id: Inference task UUID
        inference_service: Inference service dependency

    Returns:
        Current inference status

    Raises:
        HTTPException: If inference not found
    """
    try:
        # TODO: Implement status query
        # Query DCOP events for this inference_id
        # Return latest status

        logger.info(f"Querying status for inference {inference_id}")

        # Placeholder implementation
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Status query not yet implemented",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query inference status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query status: {str(e)}",
        )


# @router.post(
#     "/series/complete",
#     status_code=status.HTTP_200_OK,
#     summary="Inference completion callback",
#     description="""
#     Callback endpoint for GPU worker to report inference completion.
#
#     **Called by**: task_pipeline_inference (Series Level) after inference completes
#
#     **Payload**:
#     ```json
#     {
#       "studyInstanceUid": "study-uid",
#       "modelName": "aneurysm_model",
#       "inferenceId": "inference-uuid",
#       "result": "success",
#       "resultData": {
#         "predictions": [...]
#       }
#     }
#     ```
#
#     **Actions**:
#     - Creates SERIES_INFERENCE_COMPLETE (or FAILED) event
#     - Updates cache entry
#     - Stores result_data in DCOP event
#     """,
# )
async def inference_complete_callback(
    callback: InferenceCallbackRequest,
    inference_service: Annotated[
        DCOPEventInferenceService,
        Depends(alchemy.provide_service(DCOPEventInferenceService)),
    ],
) -> dict:
    """Handle inference completion callback from GPU worker.

    Args:
        callback: Callback payload
        inference_service: Inference service dependency

    Returns:
        Acknowledgment response

    Raises:
        HTTPException: If callback processing fails
    """
    try:
        logger.info(
            f"Received completion callback for inference {callback.inferenceId}, "
            f"result={callback.result}"
        )

        # TODO: Implement callback handling
        # 1. Create SERIES_INFERENCE_COMPLETE event (or FAILED)
        # 2. Store result_data
        # 3. Update cache

        # Placeholder implementation
        return {
            "status": "acknowledged",
            "inference_id": str(callback.inferenceId),
            "message": "Callback received (not yet fully implemented)",
        }

    except Exception as e:
        logger.error(f"Failed to process callback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process callback: {str(e)}",
        )


@router.get(
    "/cache",
    status_code=status.HTTP_200_OK,
    response_model=CacheListResponse,
    summary="List cache entries",
    description="""
    List all cached inference requests.

    Useful for monitoring and debugging cache behavior.

    **Response Example**:
    ```json
    {
      "entries": [
        {
          "cache_key": "a1b2c3d4...",
          "inference_id": "uuid",
          "study_uid": "study-uid",
          "series_uids": ["series1"],
          "model_name": "aneurysm",
          "timestamp": "2024-01-15T10:30:00",
          "status": "queued"
        }
      ],
      "total": 1
    }
    ```
    """,
)
async def list_cache(
    inference_service: Annotated[
        DCOPEventInferenceService,
        Depends(alchemy.provide_service(DCOPEventInferenceService)),
    ],
) -> CacheListResponse:
    """List all cache entries.

    Args:
        inference_service: Inference service dependency

    Returns:
        List of cache entries

    Raises:
        HTTPException: If cache listing fails
    """
    try:
        logger.info("Listing cache entries")
        return await inference_service.cache_list()

    except Exception as e:
        logger.error(f"Failed to list cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list cache: {str(e)}",
        )


@router.delete(
    "/cache",
    status_code=status.HTTP_200_OK,
    response_model=CacheDeleteResponse,
    summary="Clear cache",
    description="""
    Clear cached inference requests.

    **Warning**: This will clear ALL cache entries.
    Use for debugging or maintenance only.

    **Response Example**:
    ```json
    {
      "deleted_count": 5,
      "cache_keys": ["a1b2c3...", "d4e5f6..."]
    }
    ```
    """,
)
async def clear_cache(
    inference_service: Annotated[
        DCOPEventInferenceService,
        Depends(alchemy.provide_service(DCOPEventInferenceService)),
    ],
) -> CacheDeleteResponse:
    """Clear all cache entries.

    Args:
        inference_service: Inference service dependency

    Returns:
        Number of entries cleared

    Raises:
        HTTPException: If cache clearing fails
    """
    try:
        logger.warning("Clearing all cache entries")
        return await inference_service.cache_clear()

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )


@router.get(
    "/cache/stats",
    status_code=status.HTTP_200_OK,
    response_model=CacheStatistics,
    summary="Get cache statistics",
    description="""
    Get cache statistics for monitoring and debugging.

    **Response Example**:
    ```json
    {
      "total_entries": 15,
      "hits": 0,
      "misses": 0,
      "hit_rate": 0.0,
      "memory_bytes": 4096,
      "oldest_entry": "2024-01-15T08:00:00",
      "newest_entry": "2024-01-15T10:30:00"
    }
    ```

    **Note**: Hits/misses tracking requires application-level counters
    which are not implemented in this version. Those fields return 0.
    """,
)
async def get_cache_statistics(
    inference_service: Annotated[
        DCOPEventInferenceService,
        Depends(alchemy.provide_service(DCOPEventInferenceService)),
    ],
) -> CacheStatistics:
    """Get cache statistics.

    Args:
        inference_service: Inference service dependency

    Returns:
        Cache statistics including entry count and memory usage

    Raises:
        HTTPException: If statistics retrieval fails
    """
    try:
        logger.info("Getting cache statistics")
        return await inference_service.cache_statistics()

    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache statistics: {str(e)}",
        )


# Linus: "Good code is simple code"
# These endpoints follow REST principles:
# - POST /series → Create (queue) new inference
# - GET /series/status/{id} → Read inference status
# - POST /series/complete → Update (callback from worker)
# - GET/DELETE /cache → Cache management

# Data-driven design:
# - Request body structure determines behavior (no 'type' field needed)
# - Validation happens at Pydantic schema level (fail fast)
# - Service layer handles business logic (clean separation)
