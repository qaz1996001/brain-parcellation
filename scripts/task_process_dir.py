from code_ai.task.task_dicom2nii import dicom_to_nii
from code_ai.task.schema import intput_params
if __name__ == '__main__':
    # "raw_dicom_path": "/data/4TB1/raw_dicom/ai_team/sync_orthanc/6fabde1f-925eb6a6-2aa0fef5-2bc17b2f-6a51e3a4/12191790 HSU HUENG/20901070058 Brain Brain MRIC Health check"
    result = dicom_to_nii.push(intput_params.Dicom2NiiParams(
        sub_dir='/data/4TB1/raw_dicom/ai_team/sync_orthanc/6fabde1f-925eb6a6-2aa0fef5-2bc17b2f-6a51e3a4/12191790 HSU HUENG/20901070058 Brain Brain MRIC Health check',
        output_dicom_path='/data/4TB1/pipeline/sean/rename_dicom/12191790_20200116_MR_20901070058',
        output_nifti_path='/data/4TB1/pipeline/sean/rename_nifti/12191790_20200116_MR_20901070058',
    ).get_str_dict())
    result.set_timeout(3600)
    print(result.get())