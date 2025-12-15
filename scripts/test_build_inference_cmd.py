import argparse
import pathlib
from code_ai.utils.inference import build_inference_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        help="用於輸入的檔案",
    )
    parser.add_argument(
        "-output",
        type=str,
        help="用於輸出結果的資料夾",
    )
    args = parser.parse_args()
    nifti_study_path = pathlib.Path(args.input)
    dicom_study_path = pathlib.Path(args.output)

    inference_cmd = build_inference_cmd(nifti_study_path, dicom_study_path)
    print(inference_cmd.model_dump_json())
