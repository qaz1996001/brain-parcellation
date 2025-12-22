from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pipelinecore.core import (  # type: ignore[import-not-found]
    GpuResourceManager,
    LogManager,
    PipelineContext,
    PipelinePaths,
)


def build_runtime_components(
    pipeline_id: str,
    root_dir: Path,
    gpu_index: int = 0,
    usage_check_enabled: bool = False,
) -> Tuple[PipelineContext, GpuResourceManager]:
    """
    Convenience helper that wires up ``PipelineContext`` and ``GpuResourceManager``.
    """
    paths = PipelinePaths(
        process_dir=root_dir / "process" / pipeline_id,
        output_dir=root_dir / "output" / pipeline_id,
        log_dir=root_dir / "logs" / pipeline_id,
    )
    log_manager = LogManager(paths.log_dir, f"synthseg.{pipeline_id}")
    context = PipelineContext(
        pipeline_id=pipeline_id,
        paths=paths,
        logger=log_manager.create_logger(),
    )
    gpu_manager = GpuResourceManager(
        gpu_index=gpu_index,
        usage_check_enabled=usage_check_enabled,
    )
    return context, gpu_manager

