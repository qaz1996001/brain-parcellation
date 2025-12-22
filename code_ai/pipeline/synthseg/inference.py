from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import nibabel as nib  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from code_ai.utils.resample import (
    resampleSynthSEG2original_z_index,
    resample_one,
    save_original_seg_by_argmin_z_index,
)
from code_ai.utils_synthseg import SynthSeg, TemplateProcessor

from .models import SynthsegCasePlan, SynthsegJobResult, SynthsegPreparedBatch
from .postprocess import (
    compute_cmb,
    compute_dwi,
    compute_white_matter_masks,
    compute_wmh,
    save_volume,
)


def run_prepared_batch(
    prepared: SynthsegPreparedBatch,
    logger: logging.Logger,
) -> SynthsegJobResult:
    synth_seg = SynthSeg()
    result = SynthsegJobResult()
    for case in prepared.cases:
        try:
            if case.template_plan:
                _run_template_case(case, prepared, synth_seg, logger)
            else:
                _run_standard_case(case, prepared, synth_seg)
            result.record_success(case.source_file)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("SynthSeg case failed: %s", case.source_file, exc_info=exc)
            result.record_failure(case.source_file)
        finally:
            gc.collect()
    return result


def run_five_class_batch(
    prepared: SynthsegPreparedBatch,
    logger: logging.Logger,
) -> SynthsegJobResult:
    synth_seg = SynthSeg()
    result = SynthsegJobResult()
    for case in prepared.cases:
        try:
            resample_one(str(case.source_file), str(case.resample_file))
            synth_seg.run_segmentations5(
                path_images=str(case.resample_file),
                path_segmentations5=str(case.synthseg5_file),
            )
            resampleSynthSEG2original_z_index(
                case.source_file,
                case.resample_file,
                case.synthseg5_file,
            )
            result.record_success(case.source_file)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "SynthSeg 5-class case failed: %s", case.source_file, exc_info=exc
            )
            result.record_failure(case.source_file)
        finally:
            gc.collect()
    return result


def _run_standard_case(
    case: SynthsegCasePlan,
    prepared: SynthsegPreparedBatch,
    synth_seg: SynthSeg,
) -> None:
    resample_one(str(case.source_file), str(case.resample_file))

    synth_seg.run(
        path_images=str(case.resample_file),
        path_segmentations=str(case.synthseg_file),
        path_segmentations33=str(case.synthseg33_file),
    )
    synth_seg.run_segmentations5(
        path_images=str(case.resample_file),
        path_segmentations5=str(case.synthseg5_file),
    )

    _, argmin = resampleSynthSEG2original_z_index(
        case.source_file,
        case.resample_file,
        case.synthseg5_file,
    )

    synthseg_nii = nib.load(str(case.synthseg_file))
    synthseg_array = np.asarray(synthseg_nii.dataobj)
    synthseg33_array = np.asarray(nib.load(str(case.synthseg33_file)).dataobj)

    seg_array, synthseg_array_wm = compute_white_matter_masks(
        synthseg_array=synthseg_array,
        synthseg33_array=synthseg33_array,
        depth_number=prepared.depth_number,
    )

    if case.wm_file:
        save_volume(synthseg_array_wm, synthseg_nii, case.wm_file)
    if prepared.flags.generate_wm and case.david_file:
        save_volume(seg_array, synthseg_nii, case.david_file)

    save_original_seg_by_argmin_z_index(
        case.source_file,
        case.synthseg_file,
        argmin=argmin,
    )
    save_original_seg_by_argmin_z_index(
        case.source_file,
        case.synthseg33_file,
        argmin=argmin,
    )
    if prepared.flags.generate_wm and case.wm_file:
        save_original_seg_by_argmin_z_index(
            case.source_file,
            case.wm_file,
            argmin=argmin,
        )
    if prepared.flags.generate_wm and case.david_file:
        save_original_seg_by_argmin_z_index(
            case.source_file,
            case.david_file,
            argmin=argmin,
        )

    if prepared.flags.generate_cmb and case.cmb_file:
        cmb_array = compute_cmb(seg_array)
        save_volume(cmb_array, synthseg_nii, case.cmb_file)
        save_original_seg_by_argmin_z_index(
            case.source_file,
            case.cmb_file,
            argmin=argmin,
        )

    if prepared.flags.generate_dwi and case.dwi_file:
        dwi_array = compute_dwi(seg_array)
        save_volume(dwi_array, synthseg_nii, case.dwi_file)
        save_original_seg_by_argmin_z_index(
            case.source_file,
            case.dwi_file,
            argmin=argmin,
        )

    if prepared.flags.generate_wmh and case.wmh_file:
        wmh_array = compute_wmh(
            synthseg_array=synthseg_array,
            synthseg_array_wm=synthseg_array_wm,
            depth_number=prepared.depth_number,
        )
        save_volume(wmh_array, synthseg_nii, case.wmh_file)
        save_original_seg_by_argmin_z_index(
            case.source_file,
            case.wmh_file,
            argmin=argmin,
        )


def _run_template_case(
    case: SynthsegCasePlan,
    prepared: SynthsegPreparedBatch,
    synth_seg: SynthSeg,
    logger: logging.Logger,
) -> None:
    template_plan = case.template_plan
    if not template_plan:
        raise ValueError("Template plan expected for template-aware case.")

    resample_one(str(case.source_file), str(case.resample_file))
    synth_seg.run_segmentations33(
        path_images=str(case.resample_file),
        path_segmentations33=str(case.synthseg33_file),
    )

    resample_one(str(template_plan.template_file), str(template_plan.resample_file))
    synth_seg.run(
        path_images=str(template_plan.resample_file),
        path_segmentations=str(template_plan.synthseg_file),
        path_segmentations33=str(template_plan.synthseg33_file),
    )

    _, argmin = resampleSynthSEG2original_z_index(
        case.source_file,
        case.resample_file,
        case.synthseg33_file,
    )

    synthseg_nii = nib.load(str(template_plan.synthseg_file))
    synthseg_array = np.asarray(synthseg_nii.dataobj)
    synthseg33_array = np.asarray(nib.load(str(template_plan.synthseg33_file)).dataobj)
    seg_array, synthseg_array_wm = compute_white_matter_masks(
        synthseg_array=synthseg_array,
        synthseg33_array=synthseg33_array,
        depth_number=prepared.depth_number,
    )

    if prepared.flags.generate_wm and case.david_file:
        save_volume(seg_array, synthseg_nii, case.david_file)
    if case.wm_file:
        save_volume(synthseg_array_wm, synthseg_nii, case.wm_file)

    if prepared.flags.generate_cmb and case.cmb_file:
        cmb_array = compute_cmb(seg_array)
        save_volume(cmb_array, synthseg_nii, case.cmb_file)
    if prepared.flags.generate_dwi and case.dwi_file:
        dwi_array = compute_dwi(seg_array)
        save_volume(dwi_array, synthseg_nii, case.dwi_file)
    if prepared.flags.generate_wmh and case.wmh_file:
        wmh_array = compute_wmh(
            synthseg_array=synthseg_array,
            synthseg_array_wm=synthseg_array_wm,
            depth_number=prepared.depth_number,
        )
        save_volume(wmh_array, synthseg_nii, case.wmh_file)

    _register_template_outputs(case, prepared, argmin, logger)


def _register_template_outputs(
    case: SynthsegCasePlan,
    prepared: SynthsegPreparedBatch,
    argmin: np.ndarray,
    logger: logging.Logger,
) -> None:
    template_plan = case.template_plan
    if not template_plan:
        return

    template_basename = template_plan.synthseg33_file.stem
    synthseg_basename = case.synthseg33_file.stem
    template_coreg_file = template_plan.synthseg33_file.parent / (
        f"{synthseg_basename}_from_{template_basename}"
    )

    flirt_cmd = TemplateProcessor.flirt_cmd_base.format(
        template_plan.synthseg33_file,
        case.synthseg33_file,
        template_coreg_file,
    )
    logger.info("Running FLIRT base registration: %s", flirt_cmd)
    os.system(flirt_cmd)

    if prepared.flags.generate_cmb and case.cmb_file:
        cmb_coreg_name = case.synthseg33_file.parent / "{}_from_{}".format(
            synthseg_basename.replace("synthseg33", prepared.output_names.cmb),
            case.cmb_file.stem,
        )
        flirt_cmb_cmd = TemplateProcessor.flirt_cmd_apply.format(
            case.cmb_file,
            case.synthseg33_file,
            cmb_coreg_name,
            template_coreg_file,
        )
        logger.info("Applying FLIRT for CMB: %s", flirt_cmb_cmd)
        os.system(flirt_cmb_cmd)
        save_original_seg_by_argmin_z_index(
            case.source_file,
            Path(f"{cmb_coreg_name}.nii.gz"),
            argmin=argmin,
        )

    if prepared.flags.generate_dwi and case.dwi_file:
        dwi_coreg_name = (
            f"{case.dwi_file.with_suffix('').as_posix()}_{synthseg_basename}"
        )
        flirt_dwi_cmd = (
            rf'export FSLOUTPUTTYPE=NIFTI_GZ && flirt -in "{case.dwi_file}" '
            rf'-ref "{case.synthseg33_file}" '
            rf'-out "{dwi_coreg_name}" '
            rf'-init "{template_coreg_file}.mat" '
            r"-applyxfm -interp nearestneighbour"
        )
        logger.info("Applying FLIRT for DWI: %s", flirt_dwi_cmd)
        os.system(flirt_dwi_cmd)



