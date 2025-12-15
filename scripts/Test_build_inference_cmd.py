import pathlib
from code_ai.utils.inference import build_inference_cmd

if __name__ == "__main__":
    nifti_study_path = pathlib.Path(
        "/mnt/e/pipeline/sean/rename_nifti/10089413_20210201_MR_21002010079"
    )
    dicom_study_path = pathlib.Path(
        "/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079"
    )

    inference_cmd = build_inference_cmd(nifti_study_path, dicom_study_path)
    print(inference_cmd.model_dump_json())
