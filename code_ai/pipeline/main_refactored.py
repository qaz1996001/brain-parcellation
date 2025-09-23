"""
Refactored medical imaging pipeline following Linus-style good taste.

This module provides a clean, functional interface for processing medical images
including WMH, CMB, DWI analysis with zero special cases and maximum 3-level nesting.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

import nibabel as nib
import numpy as np

from code_ai.utils_parcellation import CMBProcess, DWIProcess, run_wmh, run_with_WhiteMatterParcellation
from code_ai.utils_synthseg import SynthSeg, TemplateProcessor
from code_ai.utils.resample import resampleSynthSEG2original_z_index, resample_one

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineType(str, Enum):
    """Pipeline types available for processing."""
    WMH = "wmh"
    CMB = "cmb"
    DWI = "dwi"
    SEGMENTATION = "segmentation"
    ALL = "all"


@dataclass
class PipelineConfig:
    """Configuration for pipeline processing."""
    input_path: Path
    output_path: Optional[Path] = None
    input_pattern: Optional[str] = None
    template_path: Optional[Path] = None
    template_pattern: Optional[str] = None
    pipeline_types: List[PipelineType] = None
    depth_number: int = 5
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        self.input_path = Path(self.input_path)
        if self.output_path:
            self.output_path = Path(self.output_path)
        if self.template_path:
            self.template_path = Path(self.template_path)
        if not self.pipeline_types:
            self.pipeline_types = [PipelineType.SEGMENTATION]


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    success: bool
    input_file: Path
    output_files: Dict[str, Path]
    error: Optional[str] = None


# Pipeline registry - data-driven design, no special cases
PIPELINE_PROCESSORS = {
    PipelineType.WMH: "process_wmh",
    PipelineType.CMB: "process_cmb",
    PipelineType.DWI: "process_dwi",
    PipelineType.SEGMENTATION: "process_segmentation",
}


def find_nifti_files(path: Path, pattern: Optional[str] = None) -> List[Path]:
    """Find NIfTI files in path with optional pattern matching."""
    if path.is_file() and path.suffix in ['.nii', '.gz']:
        return [path]
    
    files = list(path.rglob('*.nii*'))
    
    if pattern:
        files = [f for f in files if pattern in f.name]
    
    return sorted(files)


def generate_output_path(input_file: Path, suffix: str, output_dir: Optional[Path] = None) -> Path:
    """Generate output file path with suffix."""
    output_dir = output_dir or input_file.parent
    stem = input_file.name.replace('.nii.gz', '').replace('.nii', '')
    return output_dir / f"{stem}{suffix}.nii.gz"


async def process_segmentation(
    input_file: Path,
    config: PipelineConfig
) -> Dict[str, Path]:
    """Process segmentation using SynthSeg."""
    output_files = {}
    
    # Resample
    resample_path = generate_output_path(input_file, "_resample", config.output_path)
    resample_one(str(input_file), str(resample_path))
    output_files["resample"] = resample_path
    
    # Run SynthSeg
    synth_seg = SynthSeg()
    synthseg_path = generate_output_path(input_file, "_synthseg", config.output_path)
    synthseg33_path = generate_output_path(input_file, "_synthseg33", config.output_path)
    
    synth_seg.run(
        path_images=str(resample_path),
        path_segmentations=str(synthseg_path),
        path_segmentations33=str(synthseg33_path)
    )
    
    output_files["synthseg"] = synthseg_path
    output_files["synthseg33"] = synthseg33_path
    
    # Resample back to original space
    original_seg, _ = resampleSynthSEG2original_z_index(
        input_file, resample_path, synthseg33_path
    )
    output_files["original_seg"] = Path(original_seg)
    
    return output_files


async def process_wmh(
    input_file: Path,
    config: PipelineConfig,
    segmentation_results: Dict[str, Path]
) -> Dict[str, Path]:
    """Process WMH detection."""
    output_files = {}
    
    # Load segmentation results
    synthseg_nii = nib.load(str(segmentation_results["synthseg"]))
    synthseg33_nii = nib.load(str(segmentation_results["synthseg33"]))
    
    synthseg_array = np.array(synthseg_nii.dataobj)
    synthseg33_array = np.array(synthseg33_nii.dataobj)
    
    # Run white matter parcellation
    seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
        synthseg_array=synthseg_array,
        synthseg33=synthseg33_array,
        depth_number=config.depth_number
    )
    
    # Save parcellation result
    david_path = generate_output_path(input_file, "_david", config.output_path)
    out_nib = nib.Nifti1Image(seg_array, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, david_path)
    output_files["david"] = david_path
    
    # Run WMH detection
    wmh_array = run_wmh(seg_array)
    wmh_path = generate_output_path(input_file, "_WMH_PVS", config.output_path)
    out_nib = nib.Nifti1Image(wmh_array, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, wmh_path)
    output_files["wmh"] = wmh_path
    
    return output_files


async def process_cmb(
    input_file: Path,
    config: PipelineConfig,
    segmentation_results: Dict[str, Path]
) -> Dict[str, Path]:
    """Process CMB detection."""
    output_files = {}
    
    # Load segmentation
    synthseg_nii = nib.load(str(segmentation_results["synthseg"]))
    seg_array = np.array(synthseg_nii.dataobj)
    
    # Run CMB detection
    cmb_array = CMBProcess.run(seg_array)
    
    cmb_path = generate_output_path(input_file, "_CMB", config.output_path)
    out_nib = nib.Nifti1Image(cmb_array, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, cmb_path)
    output_files["cmb"] = cmb_path
    
    return output_files


async def process_dwi(
    input_file: Path,
    config: PipelineConfig,
    segmentation_results: Dict[str, Path]
) -> Dict[str, Path]:
    """Process DWI analysis."""
    output_files = {}
    
    # Load segmentation
    synthseg_nii = nib.load(str(segmentation_results["synthseg"]))
    seg_array = np.array(synthseg_nii.dataobj)
    
    # Run DWI processing
    dwi_result = DWIProcess.run(seg_array)
    
    dwi_path = generate_output_path(input_file, "_DWI", config.output_path)
    out_nib = nib.Nifti1Image(dwi_result, synthseg_nii.affine, synthseg_nii.header)
    nib.save(out_nib, dwi_path)
    output_files["dwi"] = dwi_path
    
    return output_files


# Processor function registry - no if-else chains
PROCESSORS = {
    "process_segmentation": process_segmentation,
    "process_wmh": process_wmh,
    "process_cmb": process_cmb,
    "process_dwi": process_dwi,
}


async def process_single_file(
    input_file: Path,
    config: PipelineConfig
) -> ProcessingResult:
    """Process a single file through requested pipelines."""
    try:
        output_files = {}
        
        # Always run segmentation first if other pipelines are requested
        if any(p in config.pipeline_types for p in [PipelineType.WMH, PipelineType.CMB, PipelineType.DWI]):
            segmentation_results = await process_segmentation(input_file, config)
            output_files.update(segmentation_results)
        elif PipelineType.SEGMENTATION in config.pipeline_types:
            segmentation_results = await process_segmentation(input_file, config)
            output_files.update(segmentation_results)
        else:
            segmentation_results = None
        
        # Process additional pipelines
        for pipeline_type in config.pipeline_types:
            if pipeline_type == PipelineType.SEGMENTATION:
                continue  # Already processed
            
            processor_name = PIPELINE_PROCESSORS.get(pipeline_type)
            if not processor_name:
                logger.warning(f"Unknown pipeline type: {pipeline_type}")
                continue
            
            processor = PROCESSORS[processor_name]
            if pipeline_type in [PipelineType.WMH, PipelineType.CMB, PipelineType.DWI]:
                if segmentation_results:
                    results = await processor(input_file, config, segmentation_results)
                    output_files.update(results)
            else:
                results = await processor(input_file, config)
                output_files.update(results)
        
        return ProcessingResult(
            success=True,
            input_file=input_file,
            output_files=output_files
        )
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        return ProcessingResult(
            success=False,
            input_file=input_file,
            output_files={},
            error=str(e)
        )


async def run_pipeline(config: PipelineConfig) -> List[ProcessingResult]:
    """Run pipeline on all matching files."""
    # Find input files
    input_files = find_nifti_files(config.input_path, config.input_pattern)
    
    if not input_files:
        raise ValueError(f"No NIfTI files found in {config.input_path}")
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process all files
    results = []
    for input_file in input_files:
        logger.info(f"Processing {input_file}")
        result = await process_single_file(input_file, config)
        results.append(result)
        
        if result.success:
            logger.info(f"Successfully processed {input_file}")
            for output_type, output_path in result.output_files.items():
                logger.info(f"  - {output_type}: {output_path}")
        else:
            logger.error(f"Failed to process {input_file}: {result.error}")
    
    # Summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"Processing complete: {successful}/{len(results)} files successful")
    
    return results


def create_config_from_args(args) -> PipelineConfig:
    """Create pipeline configuration from command line arguments."""
    # Determine pipeline types
    pipeline_types = []
    
    if args.all:
        pipeline_types = [PipelineType.WMH, PipelineType.CMB, PipelineType.DWI]
    else:
        if args.wmh:
            pipeline_types.append(PipelineType.WMH)
        if args.cmb:
            pipeline_types.append(PipelineType.CMB)
        if args.dwi:
            pipeline_types.append(PipelineType.DWI)
        if not pipeline_types:
            pipeline_types.append(PipelineType.SEGMENTATION)
    
    return PipelineConfig(
        input_path=args.input,
        output_path=args.output if hasattr(args, 'output') else None,
        input_pattern=args.input_name if hasattr(args, 'input_name') else None,
        template_path=args.template if hasattr(args, 'template') else None,
        template_pattern=args.template_name if hasattr(args, 'template_name') else None,
        pipeline_types=pipeline_types,
        depth_number=args.depth_number if hasattr(args, 'depth_number') else 5
    )


# Command line interface
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(
        description="Medical imaging pipeline with clean architecture"
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory or NIfTI file')
    parser.add_argument('-o', '--output',
                        help='Output directory (default: same as input)')
    parser.add_argument('--input_name',
                        help='Filter input files by name pattern')
    parser.add_argument('--template',
                        help='Template directory or file')
    parser.add_argument('--template_name',
                        help='Filter template files by name pattern')
    
    # Pipeline options
    parser.add_argument('--all', action='store_true',
                        help='Run all pipelines (WMH, CMB, DWI)')
    parser.add_argument('--wmh', '--WMH', action='store_true',
                        help='Run WMH detection')
    parser.add_argument('--cmb', '--CMB', action='store_true',
                        help='Run CMB detection')
    parser.add_argument('--dwi', '--DWI', action='store_true',
                        help='Run DWI analysis')
    
    parser.add_argument('--depth_number', type=int, default=5,
                        choices=[4, 5, 6, 7, 8, 9, 10],
                        help='White matter parcellation depth')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Run pipeline
    asyncio.run(run_pipeline(config))
