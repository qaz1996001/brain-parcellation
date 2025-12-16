from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pipelinecore.core import (  # type: ignore[import-not-found]
    BasePipeline,
    GpuResourceManager,
    PipelineContext,
    TensorflowPipelineMixin,
)

from .models import (
    SynthsegJobConfig,
    SynthsegJobResult,
    SynthsegOutputFlags,
    SynthsegPreparedBatch,
)
from .preparation import build_prepared_batch
from .inference import run_five_class_batch, run_prepared_batch

_FIVE_CLASS_FLAGS = SynthsegOutputFlags(
    generate_wm=False,
    generate_cmb=False,
    generate_dwi=False,
    generate_wmh=False,
)


class _BaseSynthsegPipeline(
    TensorflowPipelineMixin,
    BasePipeline[
        SynthsegJobConfig,
        SynthsegPreparedBatch,
        SynthsegJobResult,
        SynthsegJobResult,
    ],
):
    def __init__(
        self,
        context: PipelineContext,
        gpu_manager: GpuResourceManager,
    ) -> None:
        TensorflowPipelineMixin.__init__(self, gpu_manager)
        BasePipeline.__init__(self, context)

    def prepare(self, payload: SynthsegJobConfig) -> SynthsegPreparedBatch:
        normalized = _normalize_config(payload, self.context.paths.output_dir)
        return build_prepared_batch(normalized, self.context.paths)

    def run_inference(self, prepared: SynthsegPreparedBatch) -> SynthsegJobResult:
        return run_prepared_batch(prepared, self.context.logger)

    def postprocess(self, inference_result: SynthsegJobResult) -> SynthsegJobResult:
        return inference_result


class SynthsegPipeline(_BaseSynthsegPipeline):
    """Full SynthSeg workflow supporting WM/CMB/DWI/WMH outputs."""


class SynthsegFiveClassPipeline(_BaseSynthsegPipeline):
    """Specialized pipeline for 5-class SynthSeg outputs."""

    def prepare(self, payload: SynthsegJobConfig) -> SynthsegPreparedBatch:
        normalized = _normalize_config(
            payload,
            self.context.paths.output_dir,
            override_flags=_FIVE_CLASS_FLAGS,
        )
        return build_prepared_batch(normalized, self.context.paths)

    def run_inference(self, prepared: SynthsegPreparedBatch) -> SynthsegJobResult:
        return run_five_class_batch(prepared, self.context.logger)


def _normalize_config(
    payload: SynthsegJobConfig,
    default_output_dir: Path,
    override_flags: SynthsegOutputFlags | None = None,
) -> SynthsegJobConfig:
    normalized = payload
    if payload.output_path is None:
        normalized = replace(normalized, output_path=default_output_dir)

    if override_flags is not None:
        normalized = replace(
            normalized,
            flags=override_flags,
            run_all=False,
        )
    return normalized
