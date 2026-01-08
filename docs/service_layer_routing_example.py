"""Service layer routing example - How to use the inference parameter design.

This demonstrates how backend services should use the data-driven design
to route inference tasks to the correct queue without modifying existing code.

Linus: "Let the data structure do the talking."
"""

from typing import Dict, Any, List, Optional
from uuid import uuid4


def validate_inference_params(func_params: Dict[str, Any]) -> None:
    """Validate and enforce mutual exclusion (copied from design doc)"""
    has_study_path = 'nifti_study_path' in func_params
    has_series_uids = 'series_uids' in func_params

    if has_study_path and has_series_uids:
        raise ValueError(
            "Cannot specify both study-level (nifti_study_path) "
            "and series-level (series_uids) parameters. "
            "Choose one inference level."
        )

    if has_series_uids:
        required = ['series_uids', 'nifti_series_paths', 'model_id']
        missing = [f for f in required if f not in func_params]
        if missing:
            raise ValueError(f"Series-level inference missing: {missing}")

        if len(func_params['series_uids']) != len(func_params['nifti_series_paths']):
            raise ValueError(
                "series_uids and nifti_series_paths must have same length"
            )


# ============================================================================
# Example 1: Existing sync/service.py (NO CHANGES NEEDED)
# ============================================================================

class DCOPEventSyncService:
    """Existing service - continues to work without modification"""

    def queue_study_inference(
        self,
        study_uid: str,
        study_id: str,
        nifti_study_path: str,
        dicom_study_path: str,
    ) -> None:
        """Existing method - no changes required!"""
        from code_ai.task.task_pipeline import task_pipeline_inference

        # Build parameters exactly as before
        func_params = {
            'study_uid': study_uid,
            'study_id': study_id,
            'nifti_study_path': nifti_study_path,
            'dicom_study_path': dicom_study_path,
            'path_process': '/workspace/process',  # Could inject from config
            'upload_data_api_url': 'http://localhost:8000/api/v1',
        }

        # Push to queue - UNCHANGED
        task_pipeline_inference.push(func_params)

        # Linus: "If it ain't broke, don't fix it"
        # This code continues to work perfectly!


# ============================================================================
# Example 2: New inference/service.py (NEW FUNCTIONALITY)
# ============================================================================

class DCOPEventInferenceService:
    """New service for series-level inference"""

    def queue_series_inference(
        self,
        study_uid: str,
        study_id: str,
        series_uids: List[str],
        nifti_series_paths: List[str],
        model_id: str,
    ) -> str:
        """Queue series-level inference task

        Args:
            study_uid: Study instance UID
            study_id: Study ID
            series_uids: List of series instance UIDs to process
            nifti_series_paths: Corresponding NIFTI file paths
            model_id: Model UUID or identifier

        Returns:
            str: Inference ID for tracking

        Raises:
            ValueError: If parameters are invalid
        """
        from code_ai.task.task_series_inference import task_series_inference

        # Generate inference ID for tracking
        inference_id = f"inf-{uuid4()}"

        # Build series-level parameters
        func_params = {
            # Series-specific (NEW)
            'series_uids': series_uids,
            'nifti_series_paths': nifti_series_paths,
            'model_id': model_id,
            'inference_id': inference_id,

            # Study context (same as before)
            'study_uid': study_uid,
            'study_id': study_id,

            # Configuration (same pattern as before)
            'path_process': '/workspace/process',
            'path_json': '/workspace/json',
            'path_log': '/workspace/log',
            'upload_data_api_url': 'http://localhost:8000/api/v1',
        }

        # Validate before pushing (fail fast)
        validate_inference_params(func_params)

        # Push to NEW queue
        task_series_inference.push(func_params)

        return inference_id


# ============================================================================
# Example 3: Unified service with smart routing (OPTIONAL)
# ============================================================================

class UnifiedInferenceService:
    """Optional: Unified service that routes automatically

    Linus: "This is optional. Don't add complexity if you don't need it."

    Use this ONLY if you have use cases where you want to abstract
    the study/series distinction from the caller.
    """

    def queue_inference_auto(
        self,
        study_uid: str,
        study_id: str,
        # Study-level params (optional)
        nifti_study_path: Optional[str] = None,
        dicom_study_path: Optional[str] = None,
        # Series-level params (optional)
        series_uids: Optional[List[str]] = None,
        nifti_series_paths: Optional[List[str]] = None,
        model_id: Optional[str] = None,
    ) -> str:
        """Automatically route to correct queue based on parameters

        Linus: "Data structure drives behavior."

        Returns:
            str: Study UID (for study-level) or Inference ID (for series-level)
        """
        # Build params based on what was provided
        func_params = {
            'study_uid': study_uid,
            'study_id': study_id,
            'path_process': '/workspace/process',
            'upload_data_api_url': 'http://localhost:8000/api/v1',
        }

        # Add study-level params if provided
        if nifti_study_path:
            func_params['nifti_study_path'] = nifti_study_path
            func_params['dicom_study_path'] = dicom_study_path

        # Add series-level params if provided
        if series_uids:
            func_params['series_uids'] = series_uids
            func_params['nifti_series_paths'] = nifti_series_paths
            func_params['model_id'] = model_id
            func_params['inference_id'] = f"inf-{uuid4()}"

        # Validate (will fail if both study and series params provided)
        validate_inference_params(func_params)

        # Route based on data structure
        if 'series_uids' in func_params:
            # Series-level
            from code_ai.task.task_series_inference import task_series_inference
            task_series_inference.push(func_params)
            return func_params['inference_id']
        else:
            # Study-level
            from code_ai.task.task_pipeline import task_pipeline_inference
            task_pipeline_inference.push(func_params)
            return study_uid


# ============================================================================
# Usage Examples
# ============================================================================

def example_usage():
    """Show how services are used in practice"""

    # Example 1: Existing code continues to work
    print("=== Example 1: Existing Study-Level Inference ===")
    sync_service = DCOPEventSyncService()
    sync_service.queue_study_inference(
        study_uid='1.2.840.113619.2.55.3...',
        study_id='STUDY001',
        nifti_study_path='/data/nifti/study_001',
        dicom_study_path='/data/dicom/study_001',
    )
    print("✅ Study-level inference queued (existing behavior)")
    print()

    # Example 2: New series-level inference
    print("=== Example 2: New Series-Level Inference ===")
    inference_service = DCOPEventInferenceService()
    inference_id = inference_service.queue_series_inference(
        study_uid='1.2.840.113619.2.55.3...',
        study_id='STUDY001',
        series_uids=['1.2.3.4', '1.2.3.5'],
        nifti_series_paths=['/data/s1.nii.gz', '/data/s2.nii.gz'],
        model_id='uuid-model-123',
    )
    print(f"✅ Series-level inference queued: {inference_id}")
    print()

    # Example 3: Unified service with auto-routing
    print("=== Example 3: Unified Service (Auto-Routing) ===")
    unified_service = UnifiedInferenceService()

    # Study-level (auto-detected)
    result1 = unified_service.queue_inference_auto(
        study_uid='1.2.840.113619.2.55.3...',
        study_id='STUDY001',
        nifti_study_path='/data/nifti/study_001',
        dicom_study_path='/data/dicom/study_001',
    )
    print(f"✅ Auto-routed to study-level: {result1}")

    # Series-level (auto-detected)
    result2 = unified_service.queue_inference_auto(
        study_uid='1.2.840.113619.2.55.3...',
        study_id='STUDY001',
        series_uids=['1.2.3.4'],
        nifti_series_paths=['/data/s1.nii.gz'],
        model_id='uuid-model-123',
    )
    print(f"✅ Auto-routed to series-level: {result2}")
    print()

    # Example 4: Error case (mixing study and series params)
    print("=== Example 4: Error Handling (Fail Fast) ===")
    try:
        unified_service.queue_inference_auto(
            study_uid='1.2.840.113619.2.55.3...',
            study_id='STUDY001',
            nifti_study_path='/data/study',      # Study-level
            series_uids=['1.2.3.4'],             # Series-level - CONFLICT!
            nifti_series_paths=['/data/s1.nii'],
            model_id='model-123',
        )
    except ValueError as e:
        print(f"❌ Caught expected error: {e}")
        print("✅ Fail-fast behavior working correctly")


# ============================================================================
# Linus's Review
# ============================================================================

"""
Linus's Verdict: 🟢 Good Taste

Why this design is good:
1. ✅ Data structure drives behavior - no 'type' or 'level' field needed
2. ✅ Zero modification to existing code - backward compatibility 100%
3. ✅ Fail fast on invalid params - no silent failures
4. ✅ Simple routing logic - one if statement
5. ✅ No special cases - clear, unambiguous rules

What I would say in code review:
"This is how you extend a system properly. You didn't break anything,
you didn't add unnecessary complexity, and the data structure tells
you everything you need to know. Ship it."

Potential improvements:
- If you don't need UnifiedInferenceService, don't add it
- Keep it simple - two separate services is fine
- Only add abstraction when you have 3+ use cases, not 2
"""

if __name__ == '__main__':
    example_usage()
