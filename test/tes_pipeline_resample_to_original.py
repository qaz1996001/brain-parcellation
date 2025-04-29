from pathlib import Path


def test_pipeline_flirt():
    from code_ai.task.schema.intput_params import TaskInferenceParams
    from code_ai.task.task_synthseg import call_resample_to_original_task

    task_params = TaskInferenceParams(
        input_study_nifti_path=Path(
            '/data/10TB/sean/Suntory/rename_nifti'),
        output_study_nifti_path=Path(
            '/data/10TB/sean/Suntory/rename_nifti'))

    input_study_nifti_path = task_params.input_study_nifti_path
    output_study_nifti_path = task_params.output_study_nifti_path
    study_list = sorted(input_study_nifti_path.iterdir())
    for study_path in study_list:
        raw_file            = study_path.joinpath('T1BRAVO_AXI.nii.gz')
        resample_image_file = study_path.joinpath('T1BRAVO_AXI_resample.nii.gz')
        resample_seg_file = study_path.joinpath('T1BRAVO_AXI_resample_wm.nii.gz')
        if raw_file.exists() and resample_image_file.exists() and resample_seg_file.exists():
            func_params = {'raw_file': str(raw_file),
                           'resample_image_file': str(resample_image_file),
                           'resample_seg_file': str(resample_seg_file),
                           }
            call_resample_to_original_task.push(func_params)


if __name__ == '__main__':
    test_pipeline_flirt()
    # from funboost import AsyncResult
    # result = AsyncResult(task_id='c3fa5495-8ee7-4690-a5c5-a214607696ab')
    # print(result)
    # export PYTHONPATH=$(pwd) && python code_ai/pipeline/dicom_to_nii.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom_0328 --output_nifti /mnt/e/rename_nifti_0328