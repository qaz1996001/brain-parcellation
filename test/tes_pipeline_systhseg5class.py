from pathlib import Path

def test_pipeline_synthseg5class_tensorflow():
    from code_ai.task.schema.intput_params import TaskInferenceParams
    from code_ai.task.task_synthseg import call_pipeline_synthseg5class_tensorflow


    task_params = TaskInferenceParams(
        input_study_nifti_path=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/example_input'),
        output_study_nifti_path=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/example_input'))

    input_study_nifti_path = task_params.input_study_nifti_path
    output_study_nifti_path = task_params.output_study_nifti_path
    study_list = sorted(input_study_nifti_path.iterdir())
    for study_path in study_list:
        file_path_list = sorted(study_path.glob('T1BRAVO_AXI.nii.gz'))
        file_path_list = list(map(lambda x: str(x), file_path_list))
        func_params = {'Output_folder': str(output_study_nifti_path),
                       'ID': study_path.name,
                       'Inputs': file_path_list
                       }
        call_pipeline_synthseg5class_tensorflow.push(func_params)



if __name__ == '__main__':
    test_pipeline_synthseg5class_tensorflow()
    # from funboost import AsyncResult
    # result = AsyncResult(task_id='c3fa5495-8ee7-4690-a5c5-a214607696ab')
    # print(result)
    # export PYTHONPATH=$(pwd) && python code_ai/pipeline/dicom_to_nii.py --input_dicom /mnt/e/raw_dicom --output_dicom /mnt/e/rename_dicom_0328 --output_nifti /mnt/e/rename_nifti_0328