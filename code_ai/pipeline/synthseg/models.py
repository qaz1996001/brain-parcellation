from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class SynthsegOutputFlags:
    """Control which derivative masks are produced during SynthSeg post-processing."""

    generate_wm: bool = True
    generate_cmb: bool = False
    generate_dwi: bool = False
    generate_wmh: bool = False

    def enable_all(self) -> "SynthsegOutputFlags":
        return SynthsegOutputFlags(True, True, True, True)


@dataclass(frozen=True)
class SynthsegOutputNames:
    """Customizable suffix names for derivative outputs."""

    cmb: str = "CMB"
    dwi: str = "DWI"
    wmh: str = "WMH_PVS"


@dataclass(frozen=True)
class SynthsegJobConfig:
    """High-level configuration passed from CLI or API."""

    input_path: Path
    input_name: Optional[str]
    output_path: Optional[Path]
    template_path: Optional[Path]
    template_name: Optional[str]
    run_all: bool = False
    flags: SynthsegOutputFlags = field(default_factory=SynthsegOutputFlags)
    output_names: SynthsegOutputNames = field(default_factory=SynthsegOutputNames)
    depth_number: int = 5

    def with_all_enabled(self) -> "SynthsegJobConfig":
        return SynthsegJobConfig(
            input_path=self.input_path,
            input_name=self.input_name,
            output_path=self.output_path,
            template_path=self.template_path,
            template_name=self.template_name,
            run_all=True,
            flags=self.flags.enable_all(),
            output_names=self.output_names,
            depth_number=self.depth_number,
        )


@dataclass(frozen=True)
class TemplateCasePlan:
    """File layout for template-driven workflows (CMB/DWI requires templates)."""

    template_file: Path
    resample_file: Path
    synthseg_file: Path
    synthseg33_file: Path


@dataclass(frozen=True)
class SynthsegCasePlan:
    """Resolved per-case paths for SynthSeg execution."""

    source_file: Path
    resample_file: Path
    synthseg_file: Path
    synthseg5_file: Path
    synthseg33_file: Path
    wm_file: Path
    david_file: Path
    cmb_file: Optional[Path]
    dwi_file: Optional[Path]
    wmh_file: Optional[Path]
    template_plan: Optional[TemplateCasePlan] = None


@dataclass(frozen=True)
class SynthsegPreparedBatch:
    """Prepared plan ready to be executed by the pipeline."""

    cases: List[SynthsegCasePlan]
    flags: SynthsegOutputFlags
    output_names: SynthsegOutputNames
    depth_number: int


@dataclass
class SynthsegJobResult:
    """Execution summary for SynthSeg pipelines."""

    completed_cases: List[Path] = field(default_factory=list)
    failed_cases: List[Path] = field(default_factory=list)

    def record_success(self, path: Path) -> None:
        self.completed_cases.append(path)

    def record_failure(self, path: Path) -> None:
        self.failed_cases.append(path)
