import pathlib


if __name__ == '__main__':
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
    from code_ai.dicom2nii.convert.dicom_rename_mr_postprocess import PostProcessManager
    study_path_list = [
        # pathlib.Path('/mnt/e/rename_dicom/09721830_20170801_MR_E42947740001'),
        pathlib.Path('/mnt/e/rename_dicom/03695946_20231214_MR_21212130149')
                       ]
    post_process = PostProcessManager()
    for study_path in study_path_list:
        print('study_path:', study_path)
        post_process.post_process(study_path)
