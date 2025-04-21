from pathlib import Path


def test_nii_file_processing():
    from code_ai.task.task_dicom2nii import nii_file_processing
    study_path = Path('/mnt/d/wsl_ubuntu/pipeline/sean/process/Deep_synthseg/')
    study_path_list = sorted(study_path.iterdir())
    for study_folder_path in study_path_list:
        final_result = nii_file_processing.push({'study_folder_path':str(study_folder_path)})
        print("Final result from chain:", final_result)


if __name__ == '__main__':
    test_nii_file_processing()
