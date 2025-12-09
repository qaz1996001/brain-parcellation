import argparse


def main():
    from code_ai.utils.inference.base import check_study_mapping_inference
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='/mnt/e/rename_nifti_202505051/10516407_20231215_MR_21210200091/SWAN.nii.gz',

                        help='用於輸入的檔案')
    # parser.add_argument('--output', type=str, default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/',
    #                     help='用於輸出結果的資料夾')
    args = parser.parse_args()
    result_dict = check_study_mapping_inference(args.input)
    print(result_dict)


if __name__ == '__main__':
    main()