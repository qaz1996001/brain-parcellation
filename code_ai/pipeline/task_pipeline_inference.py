import pathlib

if __name__ == '__main__':
    from code_ai.task.task_pipeline import task_pipeline_inference

    nifti_study_str = '/mnt/e/pipeline/sean/rename_nifti/10516407_20231215_MR_21210200091'
    dicom_study_str = '/mnt/e/pipeline/sean/rename_dicom/10516407_20231215_MR_21210200091'
    nifti_study_path = pathlib.Path(nifti_study_str)
    dicom_study_path = pathlib.Path(dicom_study_str)
    task_pipeline_result = task_pipeline_inference.push({'nifti_study_path':str(nifti_study_path),
                                                         'dicom_study_path':str(dicom_study_path),
                                                         })