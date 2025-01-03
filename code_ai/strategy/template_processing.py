import gc
import os
import pathlib
import re
import sys
import traceback
from typing import List, Optional
import nibabel as nib
import numpy as np
from .base import ProcessingStrategy
from . import run_with_WhiteMatterParcellation, resample_one, SynthSeg, RequestIn,CMBProcess,DWIProcess,run_wmh


class TemplateProcessingStrategy(ProcessingStrategy):
    flirt_cmd_base = (
        'flirt -in "{0}" -ref "{1}" -out '
        '"{2}" -dof 6 -cost corratio -omat '
        '"{2}.mat" -interp nearestneighbour'
    )
    flirt_cmd_apply = (
        'flirt -in "{0}" -ref "{1}" '
        '-out "{2}" -init "{3}.mat" '
        '-applyxfm -interp nearestneighbour'
    )

    def process(self,request: RequestIn):
        pass


    # def process(self, args, file_list: List[pathlib.Path], template_file_list: Optional[List[pathlib.Path]]):
    #     synth_seg = SynthSeg()
    #     depth_number = args.depth_number or 5
    #     # CMB file SWAN
    #     for i, file in enumerate(file_list):
    #         try:
    #             # CMB BRAVO
    #             resample_file = args.resample_file_list[i]
    #             synthseg_file = args.synthseg_file_list[i]
    #             synthseg33_file = args.synthseg33_file_list[i]
    #             template_file = template_file_list[i]
    #
    #             self.resample_and_segment(synth_seg, template_file, resample_file, synthseg_file, synthseg33_file)
    #             synthseg_nii = nib.load(synthseg_file)
    #             synthseg_array = np.array(synthseg_nii.dataobj)
    #             synthseg33_nii = nib.load(synthseg33_file)
    #             synthseg33_array = np.array(synthseg33_nii.dataobj)
    #
    #             seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
    #                 synthseg_array, synthseg33_array, depth_number)
    #
    #             self.save_output_files(args, synthseg_nii, synthseg_array, seg_array, synthseg_array_wm, i)
    #
    #             self.process_with_template(
    #                 args=args, synth_seg=synth_seg, file=file, template_file=template_file,
    #                 synthseg33_file=synthseg33_file, index=i)
    #
    #         except Exception as e:
    #             log_error(file, e)
    #         gc.collect()


class NoTemplateProcessingStrategy(ProcessingStrategy):
    def process(self,request: RequestIn,model):
        resample_file   = replace_suffix(request.input_file,'_resample.nii.gz')
        synthseg_file   = replace_suffix(resample_file,'_synthseg.nii.gz')
        synthseg33_file = replace_suffix(resample_file,'_synthseg33.nii.gz')
        wm_file         = replace_suffix(resample_file, '_david.nii.gz')
        cmb_file        = replace_suffix(resample_file, '_CMB.nii.gz')
        dwi_file        = replace_suffix(resample_file, '_DWI.nii.gz')
        wmh_file        = replace_suffix(resample_file, '_WMH.nii.gz')
        resample_one(str(request.input_file), str(resample_file))
        model.run(path_images=str(resample_file), path_segmentations=str(synthseg_file),
                  path_segmentations33=str(synthseg33_file))

        synthseg_nii = nib.load(synthseg_file)
        synthseg33_nii = nib.load(synthseg33_file)
        synthseg_array = np.array(synthseg_nii.dataobj)
        synthseg33_array = np.array(synthseg33_nii.dataobj)

        seg_array, synthseg_array_wm = run_with_WhiteMatterParcellation(
            synthseg_array, synthseg33_array, request.depth_number)
        self.save(synthseg_nii,synthseg_file,seg_array,synthseg_array_wm,5,
                  )

    @classmethod
    def save(cls, synthseg_nii, synthseg_array, seg_array, synthseg_array_wm,depth_number,
             wm_file = None, cmb_file = None,  dwi_file = None,  wmh_file = None):
        if wm_file:
            cls.save_nifti(seg_array, synthseg_nii.affine, synthseg_nii.header, wm_file)
        if cmb_file:
            cmb_array = CMBProcess.run(seg_array)
            cls.save_nifti(cmb_array, synthseg_nii.affine, synthseg_nii.header, cmb_file)
        if dwi_file:
            dwi_array = DWIProcess.run(seg_array)
            cls.save_nifti(dwi_array, synthseg_nii.affine, synthseg_nii.header, dwi_file)
        if wmh_file:
            wmh_array = run_wmh(synthseg_array, synthseg_array_wm, depth_number)
            cls.save_nifti(wmh_array, synthseg_nii.affine, synthseg_nii.header, wmh_file)


    @classmethod
    def save_nifti(cls, data, affine, header, filename):
        out_nib = nib.Nifti1Image(data, affine, header)
        nib.save(out_nib, filename)


def replace_suffix(filename, new_suffix):
    pattern = r'\.nii\.gz$|\.nii$'
    return re.sub(pattern, new_suffix, filename)


def apply_transform(args, coregistration_file_name: pathlib.Path, template_synthseg33_file: Optional[pathlib.Path],
                    index: int):
    if args.cmb:
        cmb_file = args.cmb_file_list[index]
        cmb_basename = replace_suffix(cmb_file.name, '')
        cmb_coregistration_file_name = template_synthseg33_file.parent.joinpath(
            f'{template_synthseg33_file.name.replace("synthseg33.nii.gz", f"from_{cmb_basename}.nii.gz")}')
        flirt_cmd_str = (
            f'flirt -in "{cmb_file}" -ref "{template_synthseg33_file}" '
            f'-out "{cmb_coregistration_file_name}" '
            f'-init "{coregistration_file_name}.mat" '
            f'-applyxfm -interp nearestneighbour'
        )
        os.system(flirt_cmd_str)

    if args.dwi:
        dwi_file = args.dwi_file_list[index]
        dwi_basename = dwi_file.replace('.nii.gz', '')
        dwi_coregistration_file_name = f'{dwi_basename}_{synthseg33_file.replace(".nii.gz", "")}'
        flirt_cmd_str = (
            f'flirt -in "{dwi_file}" -ref "{synthseg33_file}" '
            f'-out "{dwi_coregistration_file_name}" '
            f'-init "{coregistration_file_name}.mat" '
            f'-applyxfm -interp nearestneighbour'
        )
        os.system(flirt_cmd_str)


def log_error(file, exception):
    error_class = exception.__class__.__name__
    detail = exception.args[0]
    cl, exc, tb = sys.exc_info()
    last_call_stack = traceback.extract_tb(tb)[-1]
    file_name = last_call_stack[0]
    line_num = last_call_stack[1]
    func_name = last_call_stack[2]
    errMsg = f"File \"{file_name}\", line {line_num}, in {func_name}: [{error_class}] {detail}"
    print(errMsg)
