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
import os
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
    CacheStatistics,
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

    # ==================== Redis Cache Methods ====================
    # Linus: "Simple is better than complex"
    # Cache stores serialized CacheEntry with TTL for deduplication

    async def cache_get(self, cache_key: str) -> Optional[CacheEntry]:
        """Get cache entry by key.

        Args:
            cache_key: Cache key (SHA256 hash from _generate_cache_key)

        Returns:
            CacheEntry if found, None otherwise
        """
        try:
            redis = await get_redis_client()
            full_key = f"{CACHE_KEY_PREFIX}{cache_key}"

            data = await redis.get(full_key)
            if data is None:
                logger.debug(f"Cache miss for key {cache_key[:16]}...")
                return None

            # Parse JSON to CacheEntry
            entry_dict = json.loads(data)

            # Convert string timestamps back to datetime
            if "timestamp" in entry_dict and isinstance(entry_dict["timestamp"], str):
                entry_dict["timestamp"] = datetime.datetime.fromisoformat(
                    entry_dict["timestamp"]
                )

            # Convert string UUIDs back to UUID objects
            if "inference_id" in entry_dict and isinstance(
                entry_dict["inference_id"], str
            ):
                entry_dict["inference_id"] = UUID(entry_dict["inference_id"])
            if "model_id" in entry_dict and entry_dict["model_id"]:
                entry_dict["model_id"] = UUID(entry_dict["model_id"])

            logger.debug(f"Cache hit for key {cache_key[:16]}...")
            return CacheEntry(**entry_dict)

        except Exception as e:
            logger.error(f"Cache get error for key {cache_key[:16]}...: {e}")
            return None

    async def cache_set(
        self,
        cache_key: str,
        entry: CacheEntry,
        ttl: int = DEFAULT_CACHE_TTL,
    ) -> bool:
        """Set cache entry with TTL.

        Args:
            cache_key: Cache key (SHA256 hash)
            entry: CacheEntry to store
            ttl: Time-to-live in seconds (default 24 hours)

        Returns:
            True if set successfully, False otherwise
        """
        try:
            redis = await get_redis_client()
            full_key = f"{CACHE_KEY_PREFIX}{cache_key}"

            # Serialize CacheEntry to JSON
            entry_dict = entry.model_dump()

            # Convert datetime and UUID to strings for JSON serialization
            if "timestamp" in entry_dict:
                entry_dict["timestamp"] = entry_dict["timestamp"].isoformat()
            if "inference_id" in entry_dict and entry_dict["inference_id"]:
                entry_dict["inference_id"] = str(entry_dict["inference_id"])
            if "model_id" in entry_dict and entry_dict["model_id"]:
                entry_dict["model_id"] = str(entry_dict["model_id"])

            data = json.dumps(entry_dict)

            # Set with TTL
            await redis.setex(full_key, ttl, data)

            logger.debug(f"Cache set for key {cache_key[:16]}... with TTL={ttl}s")
            return True

        except Exception as e:
            logger.error(f"Cache set error for key {cache_key[:16]}...: {e}")
            return False

    async def cache_delete(self, cache_key: str) -> bool:
        """Delete cache entry by key.

        Args:
            cache_key: Cache key to delete

        Returns:
            True if deleted, False if not found or error
        """
        try:
            redis = await get_redis_client()
            full_key = f"{CACHE_KEY_PREFIX}{cache_key}"

            result = await redis.delete(full_key)
            deleted = result > 0

            if deleted:
                logger.debug(f"Cache deleted for key {cache_key[:16]}...")
            else:
                logger.debug(f"Cache key not found for deletion: {cache_key[:16]}...")

            return deleted

        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key[:16]}...: {e}")
            return False

    async def cache_clear(self, pattern: str = "*") -> CacheDeleteResponse:
        """Clear cache entries matching pattern.

        Args:
            pattern: Glob pattern for key matching (default "*" = all)

        Returns:
            CacheDeleteResponse with deletion count and keys
        """
        try:
            redis = await get_redis_client()
            full_pattern = f"{CACHE_KEY_PREFIX}{pattern}"

            # Find all matching keys
            keys: List[str] = []
            async for key in redis.scan_iter(match=full_pattern):
                keys.append(key)

            if not keys:
                logger.info(f"No cache keys matching pattern: {pattern}")
                return CacheDeleteResponse(deleted_count=0, cache_keys=[])

            # Delete all matching keys
            deleted_count = await redis.delete(*keys)

            # Strip prefix from keys for response
            cache_keys = [key.replace(CACHE_KEY_PREFIX, "") for key in keys]

            logger.info(f"Cleared {deleted_count} cache entries matching: {pattern}")
            return CacheDeleteResponse(
                deleted_count=deleted_count, cache_keys=cache_keys
            )

        except Exception as e:
            logger.error(f"Cache clear error for pattern {pattern}: {e}")
            return CacheDeleteResponse(deleted_count=0, cache_keys=[])

    async def cache_list(self) -> CacheListResponse:
        """List all cache entries.

        Returns:
            CacheListResponse with all cache entries and total count
        """
        try:
            redis = await get_redis_client()
            full_pattern = f"{CACHE_KEY_PREFIX}*"

            entries: List[CacheEntry] = []

            async for key in redis.scan_iter(match=full_pattern):
                data = await redis.get(key)
                if data:
                    try:
                        entry_dict = json.loads(data)

                        # Convert string fields back to proper types
                        if "timestamp" in entry_dict and isinstance(
                            entry_dict["timestamp"], str
                        ):
                            entry_dict["timestamp"] = datetime.datetime.fromisoformat(
                                entry_dict["timestamp"]
                            )
                        if "inference_id" in entry_dict and isinstance(
                            entry_dict["inference_id"], str
                        ):
                            entry_dict["inference_id"] = UUID(
                                entry_dict["inference_id"]
                            )
                        if "model_id" in entry_dict and entry_dict["model_id"]:
                            entry_dict["model_id"] = UUID(entry_dict["model_id"])

                        entries.append(CacheEntry(**entry_dict))
                    except Exception as parse_error:
                        logger.warning(
                            f"Failed to parse cache entry {key}: {parse_error}"
                        )

            logger.info(f"Listed {len(entries)} cache entries")
            return CacheListResponse(entries=entries, total=len(entries))

        except Exception as e:
            logger.error(f"Cache list error: {e}")
            return CacheListResponse(entries=[], total=0)

    async def cache_check(
        self, study_uid: str, series_uids: List[str], model_id: str
    ) -> Optional[UUID]:
        """Check if inference is cached and return inference_id.

        Convenience method combining _generate_cache_key + cache_get.

        Args:
            study_uid: Study instance UID
            series_uids: List of series instance UIDs
            model_id: Model identifier

        Returns:
            UUID inference_id if cached, None otherwise
        """
        cache_key = self._generate_cache_key(study_uid, series_uids, model_id)
        entry = await self.cache_get(cache_key)

        if entry and entry.status == "completed":
            logger.info(
                f"Cache hit: inference {entry.inference_id} for "
                f"study {study_uid}, {len(series_uids)} series"
            )
            return entry.inference_id

        return None

    async def cache_store(
        self,
        study_uid: str,
        series_uids: List[str],
        model_id: str,
        inference_id: UUID,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        status: str = "queued",
    ) -> bool:
        """Store inference in cache.

        Convenience method combining _generate_cache_key + cache_set.

        Args:
            study_uid: Study instance UID
            series_uids: List of series instance UIDs
            model_id: Model identifier
            inference_id: Inference task UUID
            model_name: Optional model name
            model_version: Optional model version
            status: Cache entry status (default "queued")

        Returns:
            True if stored successfully
        """
        cache_key = self._generate_cache_key(study_uid, series_uids, model_id)

        entry = CacheEntry(
            cache_key=cache_key,
            inference_id=inference_id,
            study_uid=study_uid,
            series_uids=series_uids,
            model_name=model_name,
            model_version=model_version,
            model_id=UUID(model_id) if self._is_uuid(model_id) else None,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            status=status,
        )

        return await self.cache_set(cache_key, entry)

    async def cache_statistics(self) -> CacheStatistics:
        """Get cache statistics for monitoring.

        Returns:
            CacheStatistics with entry count, memory usage, and timestamps

        Note:
            Hits/misses tracking requires application-level counters.
            This implementation focuses on current cache state metrics.
        """
        try:
            redis = await get_redis_client()
            pattern = f"{CACHE_KEY_PREFIX}*"

            # Count entries and gather timestamps
            total_entries = 0
            memory_bytes = 0
            oldest_ts: Optional[datetime.datetime] = None
            newest_ts: Optional[datetime.datetime] = None

            async for key in redis.scan_iter(match=pattern):
                total_entries += 1

                # Get memory usage for this key
                mem = await redis.memory_usage(key)
                if mem:
                    memory_bytes += mem

                # Get entry to extract timestamp
                data = await redis.get(key)
                if data:
                    try:
                        entry_dict = json.loads(data)
                        if "timestamp" in entry_dict:
                            ts = datetime.datetime.fromisoformat(
                                entry_dict["timestamp"]
                            )
                            if oldest_ts is None or ts < oldest_ts:
                                oldest_ts = ts
                            if newest_ts is None or ts > newest_ts:
                                newest_ts = ts
                    except (json.JSONDecodeError, ValueError):
                        pass

            logger.debug(
                f"Cache statistics: {total_entries} entries, {memory_bytes} bytes"
            )

            return CacheStatistics(
                total_entries=total_entries,
                hits=0,  # Application-level tracking needed
                misses=0,  # Application-level tracking needed
                hit_rate=0.0,  # Calculated from hits/misses
                memory_bytes=memory_bytes,
                oldest_entry=oldest_ts,
                newest_entry=newest_ts,
            )

        except Exception as e:
            logger.error(f"Cache statistics error: {e}")
            return CacheStatistics(total_entries=0)

    @staticmethod
    def _is_uuid(value: str) -> bool:
        """Check if string is a valid UUID.

        Args:
            value: String to check

        Returns:
            True if valid UUID format
        """
        try:
            UUID(value)
            return True
        except (ValueError, AttributeError):
            return False

    # ==================== End Redis Cache Methods ====================

    # ==================== DWI Detection from Filesystem ====================

    def _detect_from_filesystem(
        self, study_uid: str, series_uid: str
    ) -> Optional[Tuple[str, str]]:
        """从文件系统读取 DICOM 判断类型

        Linus: "Don't trust the database, trust the filesystem"

        读取一张 DICOM，用 ConvertManager.rename_dicom_path() 判断类型。
        这重用了 Worker 的判断逻辑，保证 Backend 和 Worker 一致。

        Args:
            study_uid: Study instance UID (Orthanc ID)
            series_uid: Series instance UID (Orthanc ID)

        Returns:
            "Rename dicom" or
            None (如果非  或读取失败)
        """
        import os
        from pydicom import dcmread
        from code_ai.dicom2nii.convert.dicom_rename_mr import ConvertManager

        # 1. 推断 raw_dicom 路径
        raw_dicom_path = self._infer_raw_dicom_path(study_uid, series_uid)
        if not raw_dicom_path or not os.path.exists(raw_dicom_path):
            logger.debug(
                f"Cannot detect DWI for {series_uid}: raw_dicom not found at {raw_dicom_path}"
            )
            return None

        # 2. 读取第一张 DICOM
        try:
            dicom_files = [f for f in os.listdir(raw_dicom_path) if f.endswith(".dcm")]
            if not dicom_files:
                logger.debug(f"No DICOM files in {raw_dicom_path}")
                return None

            first_dicom = os.path.join(raw_dicom_path, dicom_files[0])
            dicom_ds = dcmread(first_dicom, stop_before_pixels=True, force=True)

            # 3. 用 ConvertManager 判断（重用 Worker 逻辑）
            # ConvertManager 需要 input/output 参数，我们用临时值
            convert_mgr   = ConvertManager(input_path=raw_dicom_path, output_path="/tmp")
            rename_result = convert_mgr.rename_dicom_path(dicom_ds)

            # 4. 返回 Rename  或 Non
            if rename_result is not '':
                study_id = convert_mgr.get_study_folder_name(dicom_ds)
                logger.info(f"Detected {rename_result} for series {series_uid}")
                return rename_result, study_id
            logger.debug(f"Series {series_uid} is not DWI (got {rename_result})")
            return None

        except Exception as e:
            logger.warning(f"Failed to detect DWI for {series_uid}: {e}", exc_info=True)
            return None

    # ==================== End DWI Detection ====================

    def _validate_nifti_exists(
        self, study_id: str , target_id: str
    ) -> Optional[str]:
        """验证 NIfTI 文件存在性

        Args:
            study_id:
            target_id: Target identifier (DWI0/DWI1000 或 series_uid)

        Returns:

            NIfTI 文件路径（如果存在），否则 None
        """
        import os

        try:
            # 构造 NIfTI 路径
            config = load_backend_config_from_env(fail_safe=True)
            nifti_base = str(config.paths.path_rename_nifti)
            nifti_path = os.path.join(nifti_base, study_id, f"{target_id}.nii.gz")

            # 验证存在性
            if os.path.exists(nifti_path) and os.path.isfile(nifti_path):
                logger.debug(f"✓ NIfTI exists: {nifti_path}")
                return nifti_path
            else:
                logger.debug(f"✗ NIfTI missing: {nifti_path}")
                return None


        except Exception as e:
            logger.warning(f"Failed to validate NIfTI for {target_id}: {e}")
            return None

    def _infer_rename_dicom_path(
        self, study_id: str, target_id: str
    ) -> Optional[str]:
        """推断 rename_dicom 路径

        Args:
            study_id:
            target_id: Target identifier (Reanme)

        Returns:
            rename_dicom 目录路径（如果存在），否则 None
        """
        import os

        try:
            # 构造 rename_dicom 路径
            config = load_backend_config_from_env(fail_safe=True)
            rename_dicom_base = str(config.paths.path_rename_dicom)
            rename_dicom_path = os.path.join(rename_dicom_base, study_id, target_id)
            return rename_dicom_path

        except Exception as e:
            logger.warning(f"Failed to infer rename_dicom for {target_id}: {e}")
            return None

    def _infer_study_id_from_config(self, study_uid: str, series_uid: str) -> str:
        """从配置推断 study_id

        Args:
            study_uid: Study instance UID
            series_uid: Series instance UID

        Returns:
            study_id (patient_id_date_modality_accession)
        """
        import os
        import re

        # 尝试从 raw_dicom 路径推断
        raw_path = self._infer_raw_dicom_path(study_uid, series_uid)
        if raw_path:
            # Path: {PATH_RAW_DICOM}/{study_id}/{series_uid}
            # Extract study_id from parent directory
            parent_dir = os.path.basename(os.path.dirname(raw_path))
            study_id_pattern = re.compile(r"^\d+_\d{8}_[A-Z]+_\d+$")
            if study_id_pattern.match(parent_dir):
                return parent_dir

        # Fallback: 使用 study_uid 前8位
        return f"unknown_{study_uid[:8]}"

    async def validate_series_ready(
        self, study_uid: str, series_uids: List[str]
    ) -> Tuple[
        List[str],  # accepted_direct_uids
        List[str],  # accepted_direct_labels
        List[str],  # accepted_convert_uids
        List[str],  # accepted_convert_labels
        List[Dict[str, str]],  # rejected
        List[str],  # nifti_paths
        List[str],  # raw_dicom_paths
        List[Optional[str]],  # rename_dicom_paths
    ]:
        """验证系列准备状态（Knuth: 精确定义每个变量）

        【数学定义】
        对于每个输入 series_uid，生成 (UID, TargetLabel) 元组：
          - DWI series: (uid, "DWI0"), (uid, "DWI1000")  // 扩展
          - 其他 series: (uid, uid)                       // 一对一

        【不变量】
        - len(accepted_direct_uids) == len(accepted_direct_labels)
        - len(accepted_convert_uids) == len(accepted_convert_labels)
        - 对于每个 label，存在唯一的 uid 使得 label ∈ expand(uid)

        Linus: "Don't trust the database, trust the filesystem"

        检查逻辑：
        1. 读取 DICOM 文件判断是否 DWI（重用 ConvertManager 逻辑）
        2. 检查文件系统：NIfTI 存在 → Direct mode，否则检查 raw_dicom → Conversion mode
        3. DWI 系列扩展为 DWI0 和 DWI1000（UID 重复，Label 不同）

        Args:
            study_uid: Study instance UID
            series_uids: List of series instance UIDs to validate (来自数据库）

        Returns:
            Tuple of:
            - accepted_direct_uids: Series UIDs with NIfTI ready (直接模式的真实 UID)
            - accepted_direct_labels: Target labels for direct mode (对应的目标标签)
            - accepted_convert_uids: Series UIDs needing conversion (转换模式的真实 UID)
            - accepted_convert_labels: Target labels for conversion mode (对应的目标标签)
            - rejected: Series not ready with reasons
            - nifti_paths: NIfTI paths for direct mode series
            - raw_dicom_paths: Raw DICOM paths for conversion mode series
            - rename_dicom_paths: Rename DICOM paths for direct mode series
        """
        import os

        # Knuth: 每个变量都精确定义其用途
        accepted_direct_uids: List[str] = []  # 真实 UID（Direct 模式）
        accepted_direct_labels: List[str] = []  # Target 标签（Direct 模式）
        accepted_convert_uids: List[str] = []  # 真实 UID（Convert 模式）
        accepted_convert_labels: List[str] = []  # Target 标签（Convert 模式）
        rejected: List[Dict[str, str]] = []
        nifti_paths: List[str] = []
        raw_dicom_paths: List[str] = []
        rename_dicom_paths: List[Optional[str]] = []

        # 【主循环：对每个 series_uid 验证并扩展】
        for series_uid in series_uids:
            # 检测是否 DWI（从文件系统读取 DICOM）
            rename_type, study_id = self._detect_from_filesystem(study_uid, series_uid)

            if rename_type in ["DWI0", "DWI1000"] :
                # 【DWI 扩展】：Series = (UID, TargetLabel)
                # expand("308454c5-...") = {"DWI0", "DWI1000"}
                targets = ["DWI0", "DWI1000"]
                logger.info(
                    f"Series {series_uid} detected as DWI, expanding to {targets}"
                )
            else:
                # 【非扩展】：Series = (UID, UID)
                # expand("86364c14-...") = {"86364c14-..."}
                targets = [rename_type]


            # 【验证每个 target】：检查文件系统是否就绪
            for target_id in targets:
                # 推断 rename_dicom 路径 (for --InputsDicomDir)
                rename_dicom_path = self._infer_rename_dicom_path(
                    study_id, target_id,
                )
                rename_dicom_paths.append(rename_dicom_path)

                # 尝试 Direct Mode: 检查 NIfTI 是否存在
                nifti_path = self._validate_nifti_exists(
                    study_id, target_id
                )

                if nifti_path:
                    # Direct mode: NIfTI 已存在
                    # 保存元组 (UID, Label)
                    accepted_direct_uids.append(series_uid)  # ✅ 真实 UID
                    accepted_direct_labels.append(target_id)  # ✅ Target 标签
                    nifti_paths.append(nifti_path)

                    logger.info(
                        f"Target {target_id} ready (direct mode), "
                        f"uid={series_uid}, nifti: {nifti_path}, rename_dicom: {rename_dicom_path}"
                    )
                    continue

                # 尝试 Conversion Mode: 检查 raw_dicom 是否存在
                raw_path = self._infer_raw_dicom_path(study_uid, series_uid)

                if raw_path and os.path.exists(raw_path):
                    # Conversion mode: 需要转换
                    # 保存元组 (UID, Label)
                    accepted_convert_uids.append(series_uid)  # ✅ 真实 UID
                    accepted_convert_labels.append(target_id)  # ✅ Target 标签
                    raw_dicom_paths.append(raw_path)
                    logger.info(
                        f"Target {target_id} ready (conversion mode), "
                        f"uid={series_uid}, raw: {raw_path} "
                        f" rename_dicom_paths={rename_dicom_paths} "
                    )
                    continue

                # 两者都不存在: 拒绝
                rejected.append(
                    {
                        "series_uid": series_uid,  # ✅ 使用 UID（数据库可查）
                        "target_id": target_id,  # 额外记录 target
                        "reason": f"Neither NIfTI nor raw DICOM found for {series_uid}",
                    }
                )
                logger.warning(
                    f"Target {target_id} (original UID: {series_uid}) not ready: "
                    f"neither NIfTI nor raw DICOM exists"
                )

        # 【验证不变量】：Knuth 的正确性检查
        assert len(accepted_direct_uids) == len(accepted_direct_labels), (
            f"Direct mode invariant violation: {len(accepted_direct_uids)} UIDs vs {len(accepted_direct_labels)} labels"
        )
        assert len(accepted_convert_uids) == len(accepted_convert_labels), (
            f"Convert mode invariant violation: {len(accepted_convert_uids)} UIDs vs {len(accepted_convert_labels)} labels"
        )

        logger.info(
            f"Validation complete: {len(accepted_direct_uids)} direct, "
            f"{len(accepted_convert_uids)} convert, {len(rejected)} rejected "
            f"(from {len(series_uids)} original series)"
        )

        return (
            accepted_direct_uids,
            accepted_direct_labels,
            accepted_convert_uids,
            accepted_convert_labels,
            rejected,
            nifti_paths,
            raw_dicom_paths,
            rename_dicom_paths,
        )

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

    def _extract_rename_dicom_path(self, event: DCOPEventModel) -> str:
        """Extract rename_dicom path from SERIES_CONVERSION_COMPLETE event.

        The SERIES_CONVERSION_COMPLETE event contains params_data with:
        {
            'output_dicom_path': '/path/to/rename_dicom/study_id/series_desc',
            ...
        }

        IMPORTANT: The stored path may be study-level. We need to ensure
        the returned path is series-level: {base_path}/{study_uid}/{series_uid}

        Args:
            event: SERIES_CONVERSION_COMPLETE event with params_data

        Returns:
            str: Path to rename_dicom series directory

        Raises:
            None - returns empty string on failure with warning
        """
        import os

        result = event.result_data or {}
        params = event.params_data or {}

        # Primary source: params_data from SERIES_CONVERSION_COMPLETE event
        rename_dicom_path = params.get("output_dicom_path", "")

        # Fallback: check result_data for alternative field names
        if not rename_dicom_path:
            rename_dicom_path = (
                result.get("rename_dicom_path")
                or result.get("output_dicom_path")
                or params.get("rename_dicom_path")
                or ""
            )

        if not rename_dicom_path:
            logger.warning(
                f"Could not extract rename_dicom_path from event {event.VsPrimaryKey}, "
                f"result_data keys: {list(result.keys())}, params_data keys: {list(params.keys())}"
            )
            return ""

        # Ensure path is series-level by appending series_uid if needed
        # Path structure should be: {PATH_RENAME_DICOM}/{study_uid}/{series_uid}
        series_uid = event.series_uid
        if series_uid and not rename_dicom_path.endswith(str(series_uid)):
            # The stored path might be study-level, append series_uid
            series_level_path = os.path.join(rename_dicom_path, str(series_uid))
            if os.path.exists(series_level_path) and os.path.isdir(series_level_path):
                logger.debug(
                    f"Converted study-level path to series-level: {rename_dicom_path} → {series_level_path}"
                )
                return series_level_path
            else:
                # Series subfolder doesn't exist, check if original path exists
                if os.path.exists(rename_dicom_path) and os.path.isdir(
                    rename_dicom_path
                ):
                    logger.debug(
                        f"Series subfolder not found: {series_level_path}, using original: {rename_dicom_path}"
                    )
                    return rename_dicom_path
                else:
                    logger.warning(
                        f"Rename DICOM path does not exist: {rename_dicom_path}"
                    )
                    return ""

        # Verify path exists before returning
        if not os.path.exists(rename_dicom_path):
            logger.warning(f"Rename DICOM path does not exist: {rename_dicom_path}")
            return ""

        return rename_dicom_path

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
            # Knuth: 明确定义 explicit mode 的数据结构
            accepted_direct_uids: List[str] = []
            accepted_direct_labels: List[str] = []
            accepted_convert_uids = request.series_uids
            accepted_convert_labels = (
                request.series_uids
            )  # 用户未提供 DWI 扩展信息，假设一对一
            rejected_series: List[Dict[str, str]] = []
            nifti_paths: List[str] = []
            raw_dicom_paths = request.raw_dicom_series_paths
            rename_dicom_paths: List[str] = []
            logger.info(
                f"Explicit conversion mode: {len(accepted_convert_uids)} series"
            )
        else:
            # Auto-detect mode: query DCOP events
            # Knuth: 使用新的 8-tuple 返回值（添加了 UIDs 和 Labels 分离）
            (
                accepted_direct_uids,
                accepted_direct_labels,
                accepted_convert_uids,
                accepted_convert_labels,
                rejected_series,
                nifti_paths,
                raw_dicom_paths,
                rename_dicom_paths,
            ) = await self.validate_series_ready(
                study_uid=request.study_uid, series_uids=request.series_uids
            )

        # Knuth: 合并 UIDs 和 Labels（保持元组关系）
        # 数学性质：len(series_uids) == len(target_labels)
        accepted_series_uids = accepted_direct_uids + accepted_convert_uids
        accepted_target_labels = accepted_direct_labels + accepted_convert_labels

        # 向后兼容：保留 accepted_series（用于 response 和日志）
        # 注意：这里仍然使用 target_labels，因为 response schema 期待可读的标签
        accepted_series = accepted_target_labels

        # Step 2: Determine model_id (resolve from name+version if needed)
        model_id = (
            str(request.model_id)
            if request.model_id
            else f"{request.model_name}:{request.model_version}"
        )

        # Step 3: Check cache (if all series accepted and direct mode only)
        cached_inference_id = None
        if (
            len(accepted_series) == len(request.series_uids)
            and not accepted_convert_uids
        ):
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
        elif len(rename_dicom_paths) > 0:
            study_id = os.path.basename(os.path.dirname(rename_dicom_paths[0]))
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

        series_uid_for_event = (
            accepted_series_uids[0] if len(accepted_series_uids) == 1 else None
        )

        # Knuth: 验证不变量（数学性质）
        assert len(accepted_series_uids) == len(accepted_target_labels), (
            f"Invariant violation: {len(accepted_series_uids)} UIDs != {len(accepted_target_labels)} labels"
        )

        # Build task parameters (Knuth: data structure drives behavior)
        func_params: Dict[str, Any] = {
            # 【Series-specific】(presence indicates series-level)
            # Knuth: 分离 UID 和 Label，满足精确性原则
            "series_uids": accepted_series_uids,  # ✅ 真实 UID（用于事件追踪）
            "target_labels": accepted_target_labels,  # ✅ Target 标签（用于文件命名和日志）
            "model_id": model_id,
            "inference_id": str(inference_id),
            # 【Study context】
            "study_uid": request.study_uid,
            "study_id": study_id,
            # 【Configuration】(dual deployment support)
            "path_process": task_paths["path_process"],
            "path_json": task_paths["path_json"],
            "path_log": task_paths["path_log"],
            "upload_data_api_url": upload_data_api_url,
        }

        # Determine mode based on what series we have
        # Knuth: "Data structure drives behavior"
        if accepted_convert_uids:
            # Conversion mode: needs DICOM → NIfTI conversion
            func_params["needs_conversion"] = True
            func_params["raw_dicom_series_paths"] = raw_dicom_paths

            # Add conversion paths if available
            if "path_rename_dicom" in task_paths:
                func_params["path_rename_dicom"] = task_paths["path_rename_dicom"]
            if "path_rename_nifti" in task_paths:
                func_params["path_rename_nifti"] = task_paths["path_rename_nifti"]

            # If we have mixed mode (some direct, some convert), include NIfTI paths too
            if accepted_direct_uids:
                func_params["nifti_series_paths"] = nifti_paths
                func_params["rename_dicom_paths"] = rename_dicom_paths
                logger.info(
                    f"Mixed mode: {len(accepted_direct_uids)} direct, {len(accepted_convert_uids)} convert"
                )
            else:
                logger.info(f"Conversion mode: {len(accepted_convert_uids)} series")
        else:
            # Direct mode: NIfTI already exists
            func_params["nifti_series_paths"] = nifti_paths
            func_params["rename_dicom_paths"] = rename_dicom_paths
            logger.info(f"Direct mode: {len(accepted_direct_uids)} series")

        # Step 6: Create SERIES_INFERENCE_READY event
        # Knuth: 事件使用真实 UID（数据库可追踪），target_labels 仅用于调试
        async with self.session_manager.get_session() as session:
            ready_event = await DCOPEventModel.create_event_ope_no(
                tool_id="SERIES_INFERENCE_TOOL",
                study_uid=request.study_uid,
                series_uid=series_uid_for_event,  # ✅ 真实 UID
                study_id=study_id,
                ope_no=DCOPStatus.SERIES_INFERENCE_READY.value,
                result_data={},  # Empty for READY event
                params_data={
                    "inference_id": str(inference_id),
                    "batch_id": str(batch_id) if batch_id else None,
                    "series_count": len(accepted_series_uids),
                    "series_uids": accepted_series_uids,  # ✅ 真实 UID（用于数据库查询）
                    "target_labels": accepted_target_labels,  # 额外记录 labels（用于调试）
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
