import pathlib


def test_pipeline():
    from code_ai.utils.inference import build_inference_cmd

    nifti_study_path = pathlib.Path('/mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/')
    dicom_study_path = pathlib.Path('/mnt/e/pipeline/sean/rename_dicom/14914694_20220905_MR_21109050071/')
    inference_cmd = build_inference_cmd(nifti_study_path,dicom_study_path)
    print(inference_cmd.model_dump_json())





if __name__ == '__main__':
    test_pipeline()