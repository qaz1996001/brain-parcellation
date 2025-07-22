from pathlib import Path

def test_pipeline_synthseg5class_tensorflow():
    from code_ai.task.task_pipeline import task_pipeline_inference

# E:\pipeline\sean\rename_dicom\10089413_20210201_MR_21002010079

    func_params = {"nifti_study_path":"/mnt/e/pipeline/sean/rename_nifti/10089413_20210201_MR_21002010079",
                   "dicom_study_path":"/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079"
                   }
    nifti_study_path = func_params['nifti_study_path']
    dicom_study_path = func_params['dicom_study_path']
    task_pipeline_inference.push(func_params)




if __name__ == '__main__':
    test_pipeline_synthseg5class_tensorflow()
    # from funboost import AsyncResult
    # result = AsyncResult(task_id='c3fa5495-8ee7-4690-a5c5-a214607696ab')
    # print(result)
    # export PYTHONPATH=$(pwd) && python code_ai/pipeline/dicom_to_nii.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom_0328 --output_nifti /mnt/e/rename_nifti_0328