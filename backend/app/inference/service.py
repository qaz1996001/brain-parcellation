"""Series-level inference service layer.

Implements the core business logic for series-level inference:
- Series validation (check if ready for inference)
- Cache management (Redis-based deduplication)
- Task queuing (push to series_inference_queue)
- Status tracking (via DCOP events)

Based on inference_params_design.md - data-driven design principles.
"""

import logging
import hashlib
from typing import List, Dict, Tuple
from uuid import UUID, uuid4

from sqlalchemy import select, and_
from advanced_alchemy.extensions.fastapi import repository

from backend.app.service import BaseRepositoryService
from backend.app.sync.schemas import DCOPStatus
from backend.app.sync.model import DCOPEventModel
from backend.app.config.api_urls import get_upload_data_api_url
from backend.app.config.task_paths import get_task_execution_paths

from .schemas import (
    SeriesInferenceRequest,
    BatchInferenceRequest,
    InferenceResponse,
    BatchInferenceResponse,
    SeriesValidationResult,
    QueueStatus,
)

logger = logging.getLogger(__name__)


class DCOPEventInferenceService(BaseRepositoryService[DCOPEventModel]):
    """Service for series-level inference orchestration."""

    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        """Repository for DCOP events."""

        model_type = DCOPEventModel

    repository_type = Repo

    def _generate_cache_key(
        self, study_uid: str, series_uids: List[str], model_id: str
    ) -> str:
        """Generate cache key from study_uid + sorted series_uids + model_id.

        Args:
            study_uid: Study instance UID
            series_uids: List of series instance UIDs
            model_id: Model identifier (UUID or name)

        Returns:
            str: SHA256 hash as cache key
        """
        # Sort series UIDs for consistent hashing
        sorted_series = sorted(series_uids)

        # Create deterministic string
        key_data = f"{study_uid}:{','.join(sorted_series)}:{model_id}"

        # Hash to fixed-length key
        cache_key = hashlib.sha256(key_data.encode()).hexdigest()

        logger.debug(
            f"Generated cache key {cache_key[:16]}... for {len(series_uids)} series"
        )
        return cache_key

    async def validate_series_ready(
        self, study_uid: str, series_uids: List[str]
    ) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
        """Validate which series are ready for inference.

        Checks DCOP events to see if series have SERIES_CONVERSION_COMPLETE status.
        Also extracts nifti file paths from the conversion complete events.

        Args:
            study_uid: Study instance UID
            series_uids: List of series instance UIDs to validate

        Returns:
            Tuple of (accepted_series, rejected_series_with_reasons, nifti_paths)
            - nifti_paths: List of nifti file paths in same order as accepted_series
        """
        accepted = []
        rejected = []
        nifti_paths = []

        async with self.session_manager.get_session() as session:
            for series_uid in series_uids:
                # Query for SERIES_CONVERSION_COMPLETE event
                stmt = (
                    select(DCOPEventModel)
                    .where(
                        and_(
                            DCOPEventModel.study_uid == study_uid,
                            DCOPEventModel.series_uid == series_uid,
                            DCOPEventModel.ope_no
                            == DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
                        )
                    )
                    .order_by(DCOPEventModel.create_time.desc())
                    .limit(1)
                )

                result = await session.execute(stmt)
                event = result.scalar_one_or_none()

                if event:
                    accepted.append(series_uid)
                    # Extract nifti path from event data
                    nifti_path = self._extract_nifti_path(event)
                    nifti_paths.append(nifti_path)
                    logger.info(
                        f"Series {series_uid} ready for inference, nifti: {nifti_path}"
                    )
                else:
                    rejected.append(
                        {
                            "series_uid": series_uid,
                            "reason": "Series conversion not complete",
                        }
                    )
                    logger.warning(
                        f"Series {series_uid} not ready: conversion not complete"
                    )

        logger.info(
            f"Validation complete: {len(accepted)} accepted, {len(rejected)} rejected "
            f"out of {len(series_uids)} total"
        )

        return accepted, rejected, nifti_paths

    def _extract_nifti_path(self, event: DCOPEventModel) -> str:
        """Extract nifti file path from SERIES_CONVERSION_COMPLETE event.

        Path is constructed from:
        - params_data['output_nifti_path']: base nifti directory
        - params_data['output_dicom_path']: contains study_id in path
        - result_data['result']: series description (filename without extension)

        Args:
            event: DCOPEventModel with conversion complete data

        Returns:
            str: Full path to the nifti file
        """
        import os

        params = event.params_data or {}
        result = event.result_data or {}

        # Get base nifti path
        output_nifti_base = params.get("output_nifti_path", "")

        # Extract study_id from output_dicom_path
        # e.g., /path/rename_dicom/10516407_20231215_MR_21210200091/T1BRAVO_AXI
        output_dicom_path = params.get("output_dicom_path", "")
        path_parts = output_dicom_path.split(os.sep)
        # Study ID is second to last part (before series description)
        study_id = path_parts[-2] if len(path_parts) >= 2 else "unknown"

        # Get series description from result
        series_desc = result.get("result", "unknown")

        # Construct full nifti path
        nifti_path = os.path.join(output_nifti_base, study_id, f"{series_desc}.nii.gz")

        return nifti_path

    async def queue_series_inference(
        self, request: SeriesInferenceRequest
    ) -> InferenceResponse:
        """Queue series-level inference task.

        Args:
            request: Series inference request

        Returns:
            InferenceResponse with inference_id and validation results
        """
        # Step 1: Validate series readiness and get nifti paths
        (
            accepted_series,
            rejected_series,
            nifti_paths,
        ) = await self.validate_series_ready(
            study_uid=request.study_uid, series_uids=request.series_uids
        )

        # Step 2: Determine model_id (resolve from name+version if needed)
        model_id = (
            str(request.model_id)
            if request.model_id
            else f"{request.model_name}:{request.model_version}"
        )

        # Step 3: Check cache (if all series accepted)
        cached_inference_id = None
        if len(accepted_series) == len(request.series_uids):
            cache_key = self._generate_cache_key(
                study_uid=request.study_uid,
                series_uids=accepted_series,
                model_id=model_id,
            )
            # TODO: Implement Redis cache check
            # cached_inference_id = await self._check_cache(cache_key)

        if cached_inference_id:
            logger.info(
                f"Cache hit for {cache_key[:16]}..., returning cached inference"
            )
            return InferenceResponse(
                study_uid=request.study_uid,
                inference_id=cached_inference_id,
                model_id=UUID(model_id) if request.model_id else None,
                model_name=request.model_name,
                model_version=request.model_version,
                series_validation=SeriesValidationResult(
                    accepted=accepted_series, rejected=rejected_series
                ),
                cache_status="cached",
                status="queued",
            )

        # Step 4: If no accepted series, return rejected status
        if not accepted_series:
            logger.warning(f"All series rejected for study {request.study_uid}")
            return InferenceResponse(
                study_uid=request.study_uid,
                inference_id=None,
                model_id=UUID(model_id) if request.model_id else None,
                model_name=request.model_name,
                model_version=request.model_version,
                series_validation=SeriesValidationResult(
                    accepted=accepted_series, rejected=rejected_series
                ),
                cache_status="new",
                status="rejected",
            )

        # Step 5: Generate inference_id and create READY event
        inference_id = uuid4()

        # Get paths from config
        task_paths = get_task_execution_paths()
        upload_data_api_url = get_upload_data_api_url()

        # Extract study_id for type safety
        study_id = f"inference_{request.study_uid[:8]}"
        series_uid_for_event = accepted_series[0] if len(accepted_series) == 1 else None

        # Build task parameters (Linus: data structure drives behavior)
        func_params = {
            # Series-specific (NEW - presence indicates series-level)
            "series_uids": accepted_series,
            "nifti_series_paths": nifti_paths,  # Paths extracted from DCOP events
            "model_id": model_id,
            "inference_id": str(inference_id),
            # Study context
            "study_uid": request.study_uid,
            "study_id": study_id,
            # Configuration (dual deployment support)
            "path_process": task_paths["path_process"],
            "path_json": task_paths["path_json"],
            "path_log": task_paths["path_log"],
            "upload_data_api_url": upload_data_api_url,
        }

        # Step 6: Create SERIES_INFERENCE_READY event
        async with self.session_manager.get_session() as session:
            ready_event = await DCOPEventModel.create_event_ope_no(
                tool_id="SERIES_INFERENCE_TOOL",
                study_uid=request.study_uid,
                series_uid=series_uid_for_event,
                study_id=study_id,
                ope_no=DCOPStatus.SERIES_INFERENCE_READY.value,
                result_data={},  # Empty for READY event
                params_data={
                    "inference_id": str(inference_id),
                    "series_count": len(accepted_series),
                    "series_uids": accepted_series,
                    "model_id": model_id,
                    "func_params": func_params,
                },
                session=session,
            )
            session.add(ready_event)
            await session.commit()

        logger.info(
            f"Created SERIES_INFERENCE_READY event for inference {inference_id}"
        )

        # Step 7: Push to unified task_pipeline_inference queue
        # Linus: Data structure drives behavior - 'series_uids' in func_params determines Series Level
        from code_ai.task.task_pipeline import task_pipeline_inference

        try:
            task_pipeline_inference.push(func_params)
            logger.info(
                f"Pushed inference {inference_id} to task_pipeline_inference_queue (Series Level)"
            )

            # Create SERIES_INFERENCE_QUEUED event
            async with self.session_manager.get_session() as session:
                queued_event = await DCOPEventModel.create_event_ope_no(
                    tool_id="SERIES_INFERENCE_TOOL",
                    study_uid=request.study_uid,
                    series_uid=series_uid_for_event,
                    study_id=study_id,
                    ope_no=DCOPStatus.SERIES_INFERENCE_QUEUED.value,
                    result_data={},  # Empty for QUEUED event
                    params_data={"inference_id": str(inference_id)},
                    session=session,
                )
                session.add(queued_event)
                await session.commit()

            # TODO: Update Redis cache with inference_id
            # await self._set_cache(cache_key, inference_id)

        except Exception as e:
            logger.error(f"Failed to queue inference {inference_id}: {e}")
            raise

        # Step 8: Determine status based on validation
        if len(accepted_series) < len(request.series_uids):
            status = "partial"  # Some series rejected
        else:
            status = "queued"  # All series accepted

        # Step 9: Return response
        return InferenceResponse(
            study_uid=request.study_uid,
            inference_id=inference_id,
            model_id=UUID(model_id) if request.model_id else None,
            model_name=request.model_name,
            model_version=request.model_version,
            series_validation=SeriesValidationResult(
                accepted=accepted_series, rejected=rejected_series
            ),
            cache_status="new",
            status=status,
            queue_status=QueueStatus(
                position=0,  # TODO: Implement queue position tracking
                estimated_wait_seconds=0,
            ),
        )

    async def process_batch_inference(
        self, batch_request: BatchInferenceRequest
    ) -> BatchInferenceResponse:
        """Process batch of series inference requests.

        Args:
            batch_request: Batch of inference requests

        Returns:
            BatchInferenceResponse with results for each request
        """
        batch_id = uuid4()
        results = []

        logger.info(
            f"Processing batch {batch_id} with {len(batch_request.requests)} requests"
        )

        for req in batch_request.requests:
            try:
                result = await self.queue_series_inference(req)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to process request for study {req.study_uid}: {e}"
                )
                # Create error response
                results.append(
                    InferenceResponse(
                        study_uid=req.study_uid,
                        inference_id=None,
                        model_id=req.model_id,
                        model_name=req.model_name,
                        model_version=req.model_version,
                        series_validation=SeriesValidationResult(
                            accepted=[],
                            rejected=[
                                {"series_uid": uid, "reason": str(e)}
                                for uid in req.series_uids
                            ],
                        ),
                        cache_status="new",
                        status="rejected",
                    )
                )

        logger.info(f"Batch {batch_id} complete: {len(results)} results")

        return BatchInferenceResponse(batch_id=batch_id, results=results)


# Linus: "Keep it simple"
# This service follows the data-driven design:
# - No 'inference_level' field needed (series_uids presence determines level)
# - Validation at service layer (fail fast)
# - Task queuing uses unified entry point (task_pipeline_inference.push)
# - DCOP events for tracking (reuse existing infrastructure)
