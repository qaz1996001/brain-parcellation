import argparse
import pathlib


def main():
    from code_ai.utils.inference import build_inference_cmd
    parser = argparse.ArgumentParser()
    parser.add_argument('--nifti_study_path', type=str,
                        default='/mnt/e/rename_nifti/10516407_20231215_MR_21210200091/',

                        help='用於輸入的檔案')
    parser.add_argument('--dicom_study_path', type=str,
                        default='/mnt/e/rename_dicom/10516407_20231215_MR_21210200091/',

                        help='用於輸入的檔案')
    # parser.add_argument('--output', type=str, default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/',
    #                     help='用於輸出結果的資料夾')
    args = parser.parse_args()
    result_dict = build_inference_cmd(pathlib.Path(args.nifti_study_path),pathlib.Path(args.dicom_study_path))
    print(result_dict)


if __name__ == '__main__':
    main()