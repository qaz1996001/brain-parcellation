import pathlib
import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
from code_ai.task.task_dicom2nii import ConvertManager



if __name__ == '__main__':
    study_folder_path = pathlib.Path('/mnt/e/rename_dicom1/07263648_20190319_MR_20803190073')
    post_process_manager = ConvertManager.dicom_post_process_manager
    post_process_manager.post_process(study_folder_path)