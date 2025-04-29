from pathlib import Path


def tes_pipeline_save_file_tasks():
    from code_ai.task.schema.intput_params import TaskInferenceParams,SaveFileTaskParams
    from code_ai.task.task_synthseg import save_file_tasks

    task_params = TaskInferenceParams(
        input_study_nifti_path=Path(
            '/data/10TB/sean/Suntory/rename_nifti'),
        output_study_nifti_path=Path(
            '/data/10TB/sean/Suntory/rename_nifti'))

    input_study_nifti_path = task_params.input_study_nifti_path
    output_study_nifti_path = task_params.output_study_nifti_path
    study_list = sorted(input_study_nifti_path.iterdir())
    for study_path in study_list:
        raw_image_file      = study_path.joinpath('T1BRAVO_AXI.nii.gz')
        resample_image_file = study_path.joinpath('T1BRAVO_AXI_resample.nii.gz')
        synthseg_file       = study_path.joinpath('synthseg_T1BRAVO_AXI_original_synthseg.nii.gz')
        synthseg33_file     = study_path.joinpath('synthseg_T1BRAVO_AXI_original_synthseg33.nii.gz')
        david_file          = study_path.joinpath('synthseg_T1BRAVO_AXI_original_david.nii.gz')
        wm_file             = study_path.joinpath('synthseg_T1BRAVO_AXI_original_wm.nii.gz')
        save_mode           = 'CMB'
        save_file_path      = study_path.joinpath('synthseg_T1BRAVO_AXI_original_CMB.nii.gz')
        task_params = SaveFileTaskParams(
            file            = raw_image_file,
            resample_file   = resample_image_file,
            synthseg_file   = synthseg_file,
            synthseg33_file = synthseg33_file,
            david_file      = david_file,
            wm_file         = wm_file,
            save_mode       = save_mode,
            save_file_path  = save_file_path,
        )
        save_file_tasks.push(task_params.get_str_dict())



if __name__ == '__main__':
    tes_pipeline_save_file_tasks()
    import tensorflow as tf
    devices = tf.config.experimental.list_physical_devices()
    # from funboost import AsyncResult
    # result = AsyncResult(task_id='c3fa5495-8ee7-4690-a5c5-a214607696ab')
    # print(result)
    # export PYTHONPATH=$(pwd) && python code_ai/pipeline/dicom_to_nii.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom_0328 --output_nifti /mnt/e/rename_nifti_0328