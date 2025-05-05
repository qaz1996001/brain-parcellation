import pathlib
import code_ai.utils_inference as utils_inference


if __name__ == '__main__':
    input_path = pathlib.Path(r'/mnt/e/rename_nifti_202505051/12472275_20231031_MR_21209070029')
    # data = utils_inference.check_study_mapping_inference(input_path)
    data = utils_inference.build_analysis(input_path)
    print(data)

