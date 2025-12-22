from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency during tests
    from pipelinecore.core import PipelinePaths  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback stub for unit tests
    class PipelinePaths:  # type: ignore[override]
        def __init__(self, process_dir: Path, output_dir: Path, log_dir: Path) -> None:
            self.process_dir = process_dir
            self.output_dir = output_dir
            self.log_dir = log_dir

from .models import (
    SynthsegCasePlan,
    SynthsegJobConfig,
    SynthsegOutputFlags,
    SynthsegOutputNames,
    SynthsegPreparedBatch,
    TemplateCasePlan,
)

_NII_SUFFIX = re.compile(r"\.nii\.gz$|\.nii$")


def build_prepared_batch(
    config: SynthsegJobConfig,
    paths: PipelinePaths,
) -> SynthsegPreparedBatch:
    """
    Resolve input/template files and materialize all intermediate/output paths.

    The function mirrors the legacy CLI pre-processing flow so we preserve the
    generated filenames while ensuring directories live under the managed
    ``PipelinePaths`` when the caller does not override ``output_path``.
    """
    input_files = _resolve_input_files(config.input_path, config.input_name)
    template_files = (
        _resolve_input_files(config.template_path, config.template_name)
        if config.template_path
        else []
    )

    if template_files and len(template_files) != len(input_files):
        raise ValueError(
            "Template file count must match input file count "
            f"(got {len(template_files)} templates for {len(input_files)} inputs)."
        )

    flags = config.flags.enable_all() if config.run_all else config.flags
    output_names = config.output_names
    target_dir = config.output_path or paths.output_dir

    cases = []
    for idx, source_file in enumerate(input_files):
        template_file = template_files[idx] if template_files else None
        case = _build_case_plan(
            idx=idx,
            source_file=source_file,
            template_file=template_file,
            flags=flags,
            output_names=output_names,
            target_dir=target_dir,
        )
        cases.append(case)

    return SynthsegPreparedBatch(
        cases=cases,
        flags=flags,
        output_names=output_names,
        depth_number=config.depth_number,
    )


def _resolve_input_files(path: Path, name_filter: str | None) -> List[Path]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.nii*"))

    if name_filter:
        files = [file for file in files if name_filter in file.name]

    if not files:
        raise ValueError(f"No NIfTI files found for {path}")

    return files


def _build_case_plan(
    idx: int,
    source_file: Path,
    template_file: Path | None,
    flags: SynthsegOutputFlags,
    output_names: SynthsegOutputNames,
    target_dir: Path | None,
) -> SynthsegCasePlan:
    resample_file = _with_suffix(source_file, "_resample.nii.gz")
    synthseg_file = _with_suffix(resample_file, "_synthseg.nii.gz")
    synthseg5_file = _with_suffix(resample_file, "_synthseg5.nii.gz")
    synthseg33_file = _with_suffix(resample_file, "_synthseg33.nii.gz")
    david_file = _with_suffix(resample_file, "_david.nii.gz")
    wm_file = _with_suffix(resample_file, "_wm.nii.gz")

    cmb_file = (
        _with_suffix(
            (template_file or resample_file),
            f"_{output_names.cmb}.nii.gz",
        )
        if flags.generate_cmb
        else None
    )
    dwi_file = (
        _with_suffix(
            (template_file or resample_file),
            f"_{output_names.dwi}.nii.gz",
        )
        if flags.generate_dwi
        else None
    )
    wmh_file = (
        _with_suffix(
            (template_file or resample_file),
            f"_{output_names.wmh}.nii.gz",
        )
        if flags.generate_wmh
        else None
    )

    template_plan = (
        _build_template_plan(template_file) if template_file else None
    )

    mapped_paths, template_plan = _maybe_redirect_outputs(
        target_dir=target_dir,
        paths=[
            resample_file,
            synthseg_file,
            synthseg5_file,
            synthseg33_file,
            david_file,
            wm_file,
            cmb_file,
            dwi_file,
            wmh_file,
        ],
        template_plan=template_plan,
    )

    (
        resample_file,
        synthseg_file,
        synthseg5_file,
        synthseg33_file,
        david_file,
        wm_file,
        cmb_file,
        dwi_file,
        wmh_file,
    ) = mapped_paths

    return SynthsegCasePlan(
        source_file=source_file,
        resample_file=resample_file,
        synthseg_file=synthseg_file,
        synthseg5_file=synthseg5_file,
        synthseg33_file=synthseg33_file,
        wm_file=wm_file,
        david_file=david_file,
        cmb_file=cmb_file,
        dwi_file=dwi_file,
        wmh_file=wmh_file,
        template_plan=template_plan,
    )


def _build_template_plan(template_file: Path) -> TemplateCasePlan:
    resample_file = _with_suffix(template_file, "_resample.nii.gz")
    synthseg_file = _with_suffix(resample_file, "_synthseg.nii.gz")
    synthseg33_file = _with_suffix(resample_file, "_synthseg33.nii.gz")
    return TemplateCasePlan(
        template_file=template_file,
        resample_file=resample_file,
        synthseg_file=synthseg_file,
        synthseg33_file=synthseg33_file,
    )


def _with_suffix(path: Path, suffix: str) -> Path:
    new_name = _NII_SUFFIX.sub(suffix, path.name)
    if new_name == path.name:
        new_name = f"{path.stem}{suffix}"
    return path.with_name(new_name)


def _maybe_redirect_outputs(
    target_dir: Path | None,
    paths: Sequence[Path | None],
    template_plan: TemplateCasePlan | None,
) -> tuple[List[Path | None], TemplateCasePlan | None]:
    if target_dir is None:
        _ensure_directories(paths, template_plan)
        return list(paths), template_plan

    redirected: List[Path | None] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path is None:
            redirected.append(None)
            continue
        redirected.append(_redirect_path(target_dir, path))

    redirected_template = template_plan
    if template_plan:
        redirected_template = TemplateCasePlan(
            template_file=template_plan.template_file,
            resample_file=_redirect_path(target_dir, template_plan.resample_file),
            synthseg_file=_redirect_path(target_dir, template_plan.synthseg_file),
            synthseg33_file=_redirect_path(target_dir, template_plan.synthseg33_file),
        )

    _ensure_directories(redirected, redirected_template)
    return redirected, redirected_template


def _ensure_directories(
    paths: Iterable[Path | None],
    template_plan: TemplateCasePlan | None,
) -> None:
    for path in paths:
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
    if template_plan:
        template_plan.resample_file.parent.mkdir(parents=True, exist_ok=True)
        template_plan.synthseg_file.parent.mkdir(parents=True, exist_ok=True)
        template_plan.synthseg33_file.parent.mkdir(parents=True, exist_ok=True)


def _redirect_path(target_dir: Path, path: Path) -> Path:
    return target_dir / f"{path.parent.name}_{path.name}"

