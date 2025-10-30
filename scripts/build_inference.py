import argparse
import pathlib

from code_ai.pipeline.rdx.schema.inference import InferenceCompleteBase
from code_ai.pipeline.rdx.schema import AneurysmDetectionResponse

from build_aneurysm import execute_rdx_platform_json


def main():
    """
    Main function to process command line arguments and execute the pipeline.

    This function:
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='01901124_20250617_MR_21404020048',
                        help='目前執行的case的patient_id or study id')

    parser.add_argument('--inputs', type=str, nargs='+',
                        default=['/mnt/e/pipeline/test_data/01901124_20250617_MR_21404020048/Image_nii/Pred.nii.gz '],
                        help='用於輸入的檔案')
    parser.add_argument('--output_folder', type=str, default='/mnt/e/pipeline/新增資料夾/01901124_20250617_MR_21404020048/output',
                        help='用於輸出結果的資料夾')
    parser.add_argument('--input_dicom_folder', type=str,
                        default='/mnt/e/pipeline/test_data/01901124_20250617_MR_21404020048/Dicom/MRA_BRAIN',
                        help='用於輸入的檔案')
    path_processModel = "/mnt/e/pipeline/test_data"
    args = parser.parse_args()

    path_root = pathlib.Path(path_processModel)
    path_processID = path_root.joinpath( args.id)
    aneurysm_platform_json :AneurysmDetectionResponse = execute_rdx_platform_json(_id=args.id,
                                                                                  path_root=path_processID)
    inference_complete = InferenceCompleteBase(study_instance_uid = aneurysm_platform_json.study_instance_uid,
                                               series_instance_uid=aneurysm_platform_json.series_instance_uid,
                                               )
    print('inference_complete',inference_complete)


if __name__ == '__main__':
    main()