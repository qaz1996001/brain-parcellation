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
import json
import datetime
from typing import List, Dict, Tuple, Any, Optional

from uuid import UUID, uuid4

from sqlalchemy import select, and_
from advanced_alchemy.extensions.fastapi import repository

from backend.app.service import BaseRepositoryService
from backend.app.sync.schemas import DCOPStatus
from backend.app.sync.model import DCOPEventModel
from backend.app.config.api_urls import get_upload_data_api_url
from backend.app.config.task_paths import get_task_execution_paths
from backend.app.config.loader import load_backend_config_from_env

from .schemas import (
    SeriesInferenceRequest,
    BatchInferenceRequest,
    InferenceResponse,
    BatchInferenceResponse,
    SeriesValidationResult,
    QueueStatus,
    CacheEntry,
    CacheListResponse,
    CacheDeleteResponse,
)
from .deps import get_redis_client, CACHE_KEY_PREFIX, DEFAULT_CACHE_TTL

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
    ) -> Tuple[List[str], List[str], List[Dict[str, str]], List[str], List[str]]:
        """Validate which series are ready for inference.

        Checks DCOP events to determine series status:
        1. SERIES_CONVERSION_COMPLETE → direct mode (NIfTI ready) - but verify file exists!
        2. SERIES_TRANSFER_COMPLETE → conversion mode (needs DICOM conversion)
        3. No events → rejected

        Linus: "Data structure drives behavior"
        Linus: "Don't trust the database blindly - verify reality"

        Args:
            study_uid: Study instance UID
            series_uids: List of series instance UIDs to validate

        Returns:
            Tuple of:
            - accepted_direct: Series with conversion complete (direct mode)
            - accepted_convert: Series needing conversion (conversion mode)
            - rejected: Series not ready with reasons
            - nifti_paths: NIfTI paths for direct mode series
            - raw_dicom_paths: Raw DICOM paths for conversion mode series
        """
        import os

        accepted_direct: List[str] = []
        accepted_convert: List[str] = []
        rejected: List[Dict[str, str]] = []
        nifti_paths: List[str] = []
        raw_dicom_paths: List[str] = []

        async with self.session_manager.get_session() as session:
            for series_uid in series_uids:
                # First, try to find SERIES_CONVERSION_COMPLETE event
                conversion_event = await self._find_event_by_status(
                    session,
                    study_uid,
                    series_uid,
                    DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
                )

                if conversion_event:
                    # Extract NIfTI path and VERIFY it exists
                    nifti_path = self._extract_nifti_path(conversion_event)

                    if nifti_path and os.path.exists(nifti_path):
                        # Direct mode: NIfTI exists and file is verified
                        accepted_direct.append(series_uid)
                        nifti_paths.append(nifti_path)
                        logger.info(
                            f"Series {series_uid} ready (direct mode), nifti: {nifti_path}"
                        )
                        continue
                    else:
                        # CONVERSION_COMPLETE event exists but file is missing!
                        # Fall through to check TRANSFER_COMPLETE for re-conversion
                        logger.warning(
                            f"Series {series_uid} has CONVERSION_COMPLETE but "
                            f"NIfTI file missing: {nifti_path}, trying conversion mode"
                        )

                # No valid conversion, try SERIES_TRANSFER_COMPLETE
                transfer_event = await self._find_event_by_status(
                    session,
                    study_uid,
                    series_uid,
                    DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
                )

                if transfer_event:
                    # Conversion mode: needs DICOM → NIfTI conversion
                    raw_path = self._extract_raw_dicom_path(transfer_event)

                    if raw_path and os.path.exists(raw_path):
                        accepted_convert.append(series_uid)
                        raw_dicom_paths.append(raw_path)
                        logger.info(
                            f"Series {series_uid} ready (conversion mode), raw: {raw_path}"
                        )
                        continue
                    else:
                        # TRANSFER_COMPLETE but raw_dicom missing
                        rejected.append(
                            {
                                "series_uid": series_uid,
                                "reason": f"Raw DICOM path missing or not accessible: {raw_path}",
                            }
                        )
                        logger.warning(
                            f"Series {series_uid} has TRANSFER_COMPLETE but "
                            f"raw DICOM missing: {raw_path}"
                        )
                        continue

                # Neither event found - try to infer raw_dicom path from config
                # Ken Thompson: "When in doubt, use brute force."
                inferred_path = self._infer_raw_dicom_path(study_uid, series_uid)

                if inferred_path:
                    # Found raw_dicom via inference from config
                    accepted_convert.append(series_uid)
                    raw_dicom_paths.append(inferred_path)
                    logger.info(
                        f"Series {series_uid} ready (inferred conversion mode), "
                        f"raw: {inferred_path}"
                    )
                    continue

                # All methods exhausted, reject
                rejected.append(
                    {
                        "series_uid": series_uid,
                        "reason": "No transfer/conversion events and could not infer raw_dicom path",
                    }
                )
                logger.warning(
                    f"Series {series_uid} not ready: no events and inference failed"
                )

        logger.info(
            f"Validation complete: {len(accepted_direct)} direct, "
            f"{len(accepted_convert)} convert, {len(rejected)} rejected "
            f"out of {len(series_uids)} total"
        )

        return accepted_direct, accepted_convert, rejected, nifti_paths, raw_dicom_paths

    async def _find_event_by_status(
        self, session, study_uid: str, series_uid: str, ope_no: str
    ):
        """Find the most recent DCOP event by status.

        Args:
            session: Database session
            study_uid: Study instance UID
            series_uid: Series instance UID
            ope_no: Operation number (status code)

        Returns:
            DCOPEventModel or None
        """
        stmt = (
            select(DCOPEventModel)
            .where(
                and_(
                    DCOPEventModel.study_uid == study_uid,
                    DCOPEventModel.series_uid == series_uid,
                    DCOPEventModel.ope_no == ope_no,
                )
            )
            .order_by(DCOPEventModel.create_time.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

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

    def _extract_raw_dicom_path(self, event: DCOPEventModel) -> str:
        """Extract raw DICOM path from SERIES_TRANSFER_COMPLETE event.

        The actual result_data format from task_dicom2nii.py is:
        {
            'raw_dicom_path': str(os.path.dirname(raw_dicom_path)),
            'rename_dicom_path': str(os.path.dirname(rename_dicom_path)),
        }

        IMPORTANT: The stored path may be study-level. We need to ensure
        the returned path is series-level: {base_path}/{series_uid}

        Args:
            event: DCOPEventModel with transfer complete data

        Returns:
            str: Path to the raw DICOM series directory
        """
        import os

        result = event.result_data or {}
        params = event.params_data or {}

        # Primary source: result_data from task_dicom2nii.py
        raw_path = result.get("raw_dicom_path", "")

        # Fallback: check params_data for alternative field names
        if not raw_path:
            raw_path = (
                params.get("raw_dicom_path")
                or params.get("input_dicom_path")
                or params.get("dicom_path")
                or ""
            )

        if not raw_path:
            logger.warning(
                f"Could not extract raw_dicom_path from event {event.VsPrimaryKey}, "
                f"result_data keys: {list(result.keys())}, params_data keys: {list(params.keys())}"
            )
            return ""

        # Ensure path is series-level by appending series_uid if needed
        # Path structure should be: {PATH_RAW_DICOM}/{study_uid}/{series_uid}
        series_uid = event.series_uid
        if series_uid and not raw_path.endswith(series_uid):
            # The stored path is study-level, append series_uid
            series_level_path = os.path.join(raw_path, str(series_uid))
            if os.path.exists(series_level_path) and os.path.isdir(series_level_path):
                logger.debug(
                    f"Converted study-level path to series-level: {raw_path} → {series_level_path}"
                )
                return series_level_path
            else:
                # Series subfolder doesn't exist, return original path
                logger.debug(
                    f"Series subfolder not found: {series_level_path}, using original: {raw_path}"
                )

        return raw_path

    def _infer_raw_dicom_path(self, study_uid: str, series_uid: str) -> str:
        """Infer raw DICOM path from configuration.

        Ken Thompson: "When in doubt, use brute force."

        When no TRANSFER_COMPLETE event exists, we can infer the raw_dicom
        path from the configured PATH_RAW_DICOM base directory.

        Path structure: {PATH_RAW_DICOM}/{study_uid}/{series_uid}

        Args:
            study_uid: Study instance UID (Orthanc ID format)
            series_uid: Series instance UID (Orthanc ID format)

        Returns:
            str: Inferred path to the raw DICOM series directory, or empty string if
                 the path doesn't exist
        """
        import os

        try:
            config = load_backend_config_from_env(fail_safe=True)
            raw_dicom_base = str(config.paths.path_raw_dicom)

            # Infer path: {PATH_RAW_DICOM}/{study_uid}/{series_uid}
            inferred_path = os.path.join(raw_dicom_base, study_uid, series_uid)

            if os.path.exists(inferred_path) and os.path.isdir(inferred_path):
                logger.info(f"Inferred raw_dicom_path from config: {inferred_path}")
                return inferred_path
            else:
                logger.debug(f"Inferred path does not exist: {inferred_path}")
                return ""
        except Exception as e:
            logger.warning(f"Failed to infer raw_dicom_path: {e}")
            return ""

    def _infer_study_id_from_paths(self, paths: List[str]) -> str:
        """Infer study_id from NIfTI or DICOM paths.

        Study ID format: {patient_id}_{study_date}_{modality}_{accession_number}
        Example: 10516407_20231215_MR_21210200091

        Path structure: {base}/{study_id}/{series_description}.nii.gz

        Args:
            paths: List of file paths (NIfTI or DICOM directories)

        Returns:
            str: Inferred study_id, or empty string if cannot infer
        """
        import os
        import re

        # Pattern for study_id: patientId_date_modality_accessionNumber
        study_id_pattern = re.compile(r"^\d+_\d{8}_[A-Z]+_\d+$")

        for path in paths:
            if not path:
                continue

            # For NIfTI files: /path/to/10516407_20231215_MR_21210200091/SWAN.nii.gz
            # For DICOM dirs: /path/to/10516407_20231215_MR_21210200091/SWAN
            parts = path.rstrip(os.sep).split(os.sep)

            # Check parent folder (second to last part)
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i]
                if study_id_pattern.match(part):
                    logger.debug(f"Inferred study_id '{part}' from path: {path}")
                    return part

        logger.warning(f"Could not infer study_id from paths: {paths[:3]}...")
        return ""

    async def _find_study_id_from_events(
        self, study_uid: str, series_uids: List[str]
    ) -> str:
        """Find study_id from DCOP events for the given series.

        Args:
            study_uid: Study instance UID
            series_uids: List of series UIDs to check

        Returns:
            str: study_id from events, or empty string if not found
        """
        async with self.session_manager.get_session() as session:
            for series_uid in series_uids:
                # Try SERIES_CONVERSION_COMPLETE first
                event = await self._find_event_by_status(
                    session,
                    study_uid,
                    series_uid,
                    DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
                )

                if event and event.params_data:
                    # Extract from output_dicom_path
                    output_dicom_path = event.params_data.get("output_dicom_path", "")
                    if output_dicom_path:
                        study_id = self._infer_study_id_from_paths([output_dicom_path])
                        if study_id:
                            return study_id

                # Try SERIES_TRANSFER_COMPLETE
                event = await self._find_event_by_status(
                    session,
                    study_uid,
                    series_uid,
                    DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
                )

                if event and event.result_data:
                    raw_path = event.result_data.get("raw_dicom_path", "")
                    if raw_path:
                        study_id = self._infer_study_id_from_paths([raw_path])
                        if study_id:
                            return study_id

        return ""

    async def queue_series_inference(
        self, request: SeriesInferenceRequest, batch_id: UUID | None = None
    ) -> InferenceResponse:
        """Queue series-level inference task.

        Supports two modes:
        1. Direct mode: Series already converted, uses NIfTI paths
        2. Conversion mode: needs_conversion=True, triggers DICOM conversion flow

        Linus: "Data structure drives behavior"

        Args:
            request: Series inference request

        Returns:
            InferenceResponse with inference_id and validation results
        """
        # Step 1: Handle explicit conversion mode (user-provided paths)
        if request.needs_conversion and request.raw_dicom_series_paths:
            # User explicitly requested conversion with paths
            accepted_direct: List[str] = []
            accepted_convert = request.series_uids
            rejected_series: List[Dict[str, str]] = []
            nifti_paths: List[str] = []
            raw_dicom_paths = request.raw_dicom_series_paths
            logger.info(f"Explicit conversion mode: {len(accepted_convert)} series")
        else:
            # Auto-detect mode: query DCOP events
            (
                accepted_direct,
                accepted_convert,
                rejected_series,
                nifti_paths,
                raw_dicom_paths,
            ) = await self.validate_series_ready(
                study_uid=request.study_uid, series_uids=request.series_uids
            )

        # Combine accepted series for response
        accepted_series = accepted_direct + accepted_convert

        # Step 2: Determine model_id (resolve from name+version if needed)
        model_id = (
            str(request.model_id)
            if request.model_id
            else f"{request.model_name}:{request.model_version}"
        )

        # Step 3: Check cache (if all series accepted and direct mode only)
        cached_inference_id = None
        if len(accepted_series) == len(request.series_uids) and not accepted_convert:
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

        # Resolve study_id with priority:
        # 1. Request provides explicit study_id
        # 2. Infer from nifti_paths (direct mode)
        # 3. Infer from raw_dicom_paths (conversion mode)
        # 4. Query DCOP events
        # 5. Fallback to placeholder (not recommended)
        study_id: str = ""

        if request.study_id:
            study_id = request.study_id
            logger.debug(f"Using study_id from request: {study_id}")
        elif nifti_paths:
            study_id = self._infer_study_id_from_paths(nifti_paths)
            if study_id:
                logger.debug(f"Inferred study_id from nifti_paths: {study_id}")
        elif raw_dicom_paths:
            study_id = self._infer_study_id_from_paths(raw_dicom_paths)
            if study_id:
                logger.debug(f"Inferred study_id from raw_dicom_paths: {study_id}")

        if not study_id:
            # Fallback: query DCOP events
            study_id = await self._find_study_id_from_events(
                request.study_uid, request.series_uids
            )
            if study_id:
                logger.debug(f"Found study_id from DCOP events: {study_id}")

        if not study_id:
            # Last resort fallback (should rarely happen)
            study_id = f"unknown_{request.study_uid[:8]}"
            logger.warning(
                f"Could not determine study_id, using fallback: {study_id}. "
                f"Consider providing study_id in request."
            )

        series_uid_for_event = accepted_series[0] if len(accepted_series) == 1 else None

        # Build task parameters (Linus: data structure drives behavior)
        func_params: Dict[str, Any] = {
            # Series-specific (presence indicates series-level)
            "series_uids": accepted_series,
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

        # Determine mode based on what series we have
        # Linus: "Data structure drives behavior"
        if accepted_convert:
            # Conversion mode: needs DICOM → NIfTI conversion
            func_params["needs_conversion"] = True
            func_params["raw_dicom_series_paths"] = raw_dicom_paths

            # Add conversion paths if available
            if "path_rename_dicom" in task_paths:
                func_params["path_rename_dicom"] = task_paths["path_rename_dicom"]
            if "path_rename_nifti" in task_paths:
                func_params["path_rename_nifti"] = task_paths["path_rename_nifti"]

            # If we have mixed mode (some direct, some convert), include NIfTI paths too
            if accepted_direct:
                func_params["nifti_series_paths"] = nifti_paths
                logger.info(
                    f"Mixed mode: {len(accepted_direct)} direct, {len(accepted_convert)} convert"
                )
            else:
                logger.info(f"Conversion mode: {len(accepted_convert)} series")
        else:
            # Direct mode: NIfTI already exists
            func_params["nifti_series_paths"] = nifti_paths
            logger.info(f"Direct mode: {len(accepted_direct)} series")

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
                    "batch_id": str(batch_id) if batch_id else None,
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
                result = await self.queue_series_inference(req, batch_id=batch_id)
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
