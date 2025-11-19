from pathlib import Path

def test_pipeline_synthseg5class_tensorflow():
    from code_ai.task.schema.intput_params import TaskInferenceParams
    from code_ai.task.task_CMB import call_pipeline_cmb_tensorflow
    # from code_ai.task.task_synthseg import call_pipeline_synthseg5class_tensorflow

    task_params = TaskInferenceParams(
        input_study_nifti_path=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/example_input'),
        output_study_nifti_path=Path(
            '/mnt/d/wsl_ubuntu/pipeline/sean/example_output'))

    input_study_nifti_path = task_params.input_study_nifti_path
    output_study_nifti_path = task_params.output_study_nifti_path
    study_list = sorted(input_study_nifti_path.iterdir())
    for study_path in study_list:

        file_path_list = sorted(study_path.glob('T1FLAIR_AXI.nii.gz')) + sorted(study_path.glob('SWAN.nii.gz'))
        print(file_path_list)
        file_path_list = list(map(lambda x: str(x), file_path_list))
        func_params = {'Output_folder': str(output_study_nifti_path),
                       'ID': study_path.name,
                       'Inputs': file_path_list
                       }
        call_pipeline_cmb_tensorflow.push(func_params)




if __name__ == '__main__':
    test_pipeline_synthseg5class_tensorflow()