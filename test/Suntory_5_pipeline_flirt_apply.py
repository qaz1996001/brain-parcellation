from pathlib import Path

from code_ai.utils.inference import replace_suffix


def test_pipeline_flirt():
    from code_ai.task.schema.intput_params import TaskInferenceParams
    from code_ai.task.task_synthseg import call_pipeline_flirt

    task_params = TaskInferenceParams(
        input_study_nifti_path=Path(
            '/data/10TB/sean/Suntory/rename_nifti'),
        output_study_nifti_path=Path(
            '/data/10TB/sean/Suntory/rename_nifti'))

    input_study_nifti_path = task_params.input_study_nifti_path
    output_study_nifti_path = task_params.output_study_nifti_path
    study_list = sorted(input_study_nifti_path.iterdir())
    for study_path in study_list:
        input_file         = study_path.joinpath('synthseg_ASLPROD_original_synthseg5.nii.gz')
        template_file      = study_path.joinpath('synthseg_T1BRAVO_AXI_original_CMB.nii.gz')
        coregistration_mat = study_path.joinpath('synthseg_ASLPROD_original_synthseg5_from_synthseg_T1BRAVO_AXI_original_synthseg5.mat')
        output_coregistration_path = study_path.joinpath('synthseg_ASLPROD_original_CMB_from_synthseg_T1BRAVO_AXI_original_CMB.nii.gz')
        if output_coregistration_path.exists():
            continue
        if input_file.exists() and template_file.exists() and coregistration_mat.exists():
            func_params = {'input_file': str(input_file),
                           'template_file': str(template_file),
                           'coregistration_mat' :str(coregistration_mat)
                           }
            call_pipeline_flirt.push(func_params)


if __name__ == '__main__':
    test_pipeline_flirt()
    # from funboost import AsyncResult
    # result = AsyncResult(task_id='c3fa5495-8ee7-4690-a5c5-a214607696ab')
    # print(result)
    # export PYTHONPATH=$(pwd) && python code_ai/pipeline/dicom_to_nii.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom_0328 --output_nifti /mnt/e/rename_nifti_0328