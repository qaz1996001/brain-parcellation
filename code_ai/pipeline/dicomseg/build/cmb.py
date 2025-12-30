import json
import pathlib
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
import pydicom
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
from typing_extensions import Self

from code_ai.pipeline import pipeline_parser
from code_ai.pipeline.dicomseg import utils
from code_ai.pipeline.dicomseg.schema import CmbDetectionItem, CmbDetectionResponse
from code_ai.pipeline.dicomseg.build.base import PredictionBaseBuilder


class CmbDetectionBuilder(PredictionBaseBuilder[CmbDetectionItem, CmbDetectionResponse]):
    """
    CMB 檢測結果 Builder
    """

    model_class = CmbDetectionResponse

    @staticmethod
    def use_create_dicom_seg_file(path_nii: pathlib.Path,
                                  series_name: str,
                                  output_folder: pathlib.Path,
                                  image: Any,
                                  first_dcm: FileDataset | DicomDir,
                                  source_images: List[FileDataset | DicomDir], ):
        # Load prediction data from NIfTI file
        if series_name == "SWAN":
            new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii)
            dicom_seg_series_name = 'Pred_CMB'
        else:
            return []

        pred_data_unique = np.unique(new_nifti_array)
        if len(pred_data_unique) < 1:
            return None
        else:
            pred_data_unique = pred_data_unique[1:]  # Exclude background value (0)
        result_list = utils.create_dicom_seg_file(pred_data_unique,
                                                  new_nifti_array,
                                                  dicom_seg_series_name,
                                                  output_folder,
                                                  image,
                                                  first_dcm,
                                                  source_images)
        return result_list

    @staticmethod
    def merge_pred_json_dicom_seg_result_list(pred_json_list:List[Dict[str, Any]],dicom_seg_result_list:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merge_pred_json = [
            {
                'series_name': pred['series_name'],
                'data': [
                    {**pred_item, **seg_item}
                    for pred_item, seg_item in zip(pred['data'], seg['data'])
                ]
            }
            for pred, seg in zip(pred_json_list, dicom_seg_result_list)
        ]
        return merge_pred_json

    def set_patient_info(self, source_images: List[Union[FileDataset, DicomDir]]) -> Self:
        """從 DICOM 提取患者基本資訊"""
        if not source_images:
            raise ValueError("source_images cannot be empty")

        dicom_ds = source_images[0]

        # 提取患者和檢查資訊
        patient_id = dicom_ds.get((0x0010, 0x0020)).value
        study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
        self._prediction_dict.update({
            'patient_id': patient_id,
            'input_study_instance_uid': [study_instance_uid],
        })
        input_series_instance_uid_list = [dicom_ds.get((0x0020, 0x000E)).value for dicom_ds in source_images]
        self._prediction_dict.update({
            'patient_id': patient_id,
            'input_series_instance_uid': input_series_instance_uid_list,
        })
        return self



    def build_detection(self, source_image: Union[FileDataset, DicomDir], prediction_result: Dict[str, Any], *args,
                        **kwargs) -> CmbDetectionItem:
        # 從 DICOM 提取必要資訊
        series_instance_uid = source_image.get((0x0020, 0x000E)).value
        sop_instance_uid = source_image.get((0x0008, 0x0018)).value
        # pred_result {'label#': 1, 'class_name': 'CMB', 'review_class_name': None,
        # 'type': 'C103', 'type_name': 'left basal ganglion', 'LPS_coordinates': None,
        # 'pred_diameter': 2.1973499805, 'CAD_method': None,
        # 'complete': None, 'Doctor': None, 'c-time': '2025/12/25 10:12', 'Reviewer': None, 'r-time': None, 'CMB_prob': 0.846863687}
        # 構建 detection 字典
        detection_dict = {
            "annotated_series_instance_uid":self._prediction_dict.get('input_series_instance_uid')[0],
            'series_instance_uid': series_instance_uid,
            'sop_instance_uid': sop_instance_uid,
            'label': f"A{prediction_result.get('mask_index', 'cmb')}",
            'type': prediction_result['class_name'],
            'location': prediction_result['type_name'],
            'diameter': prediction_result['pred_diameter'],
            'main_seg_slice': prediction_result['main_seg_slice'],
            'probability': prediction_result['CMB_prob'],
            'mask_index': prediction_result['label#'],
            'sub_location': prediction_result.get('sub_location', ""),
        }
        cmb_detection_item = CmbDetectionItem.model_validate(detection_dict)
        return cmb_detection_item



def _create_directory(directory_path :pathlib.Path):
    if directory_path.is_dir():
        directory_path.mkdir(exist_ok=True, parents=True)
    else:
        directory_path.parent.mkdir(parents=True, exist_ok=True)
    return directory_path.exists()




def main(model_id: str = '48c0cfa2-347b-4d32-aa74-a7b1e20dd2e6'):
    parser = pipeline_parser()
    args = parser.parse_args()

    # Extract arguments
    _id = args.ID
    path_dcms = pathlib.Path(args.InputsDicomDir)
    path_nii = pathlib.Path(args.Inputs[0])
    path_dcmseg = pathlib.Path(args.Output_folder)

    output_series_folder = path_dcmseg.joinpath(f'{_id}')
    _create_directory(output_series_folder)

    # Load prediction data from NIfTI file
    pred_json_path = path_nii.parent.joinpath(path_nii.name.replace('.nii.gz',
                                                                    '.json'))
    with open(pred_json_path) as f:
        pred_json = json.load(f)
    # 初始化動脈瘤檢測 JSON 建構器
    # platform_json_builder = AneurysmDetectionBuilder()
    platform_json_builder = CmbDetectionBuilder()

    # 定義序列名稱：SWAN
    series_name_list = ['SWAN', 'T1BRAVO_AXI' ,'T1FLAIR_AXI']
    path_dcms.exists()
    # 建立每個序列的 DICOM 資料夾路徑列表
    dcms_folder_path_list = list(filter(lambda x:x.exists(),
                                        [path_dcms if path_dcms.name==x else path_dcms.joinpath(x)
                                         for x in series_name_list]))

    # 建立預測 NIfTI 檔案路徑列表
    pred_nii_path_list = [{"series_name": x,"pred_nii_path": str(path_nii)}
                          for x in series_name_list
                          ]
    #              載入並排序每個序列的 DICOM 檔案
    #             - sorted_dcms: List of sorted DICOM file paths
    #             - image: SimpleITK image object
    #             - first_dcm: First DICOM dataset
    #             - source_images: List of all DICOM datasets (without pixel data)
    print('dcms_folder_path_list',dcms_folder_path_list)
    dicom_data_list = [utils.load_and_sort_dicom_files(x) for x in dcms_folder_path_list]

    # 提取每個序列的第一個 DICOM 檔案（索引 [2] 表示排序後的第一個檔案 first_dcm）
    series_first_dcm_data_list = [x[2] for x in dicom_data_list]

    # 為每個序列生成 DICOM-SEG 檔案和預測結果
    pred_result_list = [{"series_name": x[0],
                         "data": platform_json_builder.use_create_dicom_seg_file(
                             path_nii,
                             x[0],                     # sorted_dcms
                             output_series_folder,
                             *x[1][1:]  # 解包 DICOM 資料（跳過第一個元素）
                         )}
                        for x in zip(series_name_list, dicom_data_list)]

    # 讀取所有生成的 DICOM-SEG 檔案
    dcm_seg_path_list = [list(map(lambda xx: pydicom.read_file(xx['dcm_seg_path']), x['data']))
                         for x in pred_result_list]
    with open(pred_json_path) as f:
        pred_json = json.load(f)

    pred_json_list = [{"series_name": "SWAN",
                              "data": pred_json},]


    pred_json_merge = platform_json_builder.merge_pred_json_dicom_seg_result_list(pred_json_list=pred_json_list,
                                                                                  dicom_seg_result_list=pred_result_list)
    #
    #
    # # 使用建構器模式組裝最終的平台 JSON

    platform_json = (platform_json_builder
                              .set_patient_info(series_first_dcm_data_list)                     # 設定患者資訊
                              .set_model_id(model_id)                                         # 設定模型 ID
                              .set_detections(dcm_seg_path_list[0], pred_json_merge[0]['data'])  # 設定檢測結果
                              .build()  # 建構最終 JSON
                              )

    print('platform_json',platform_json)

    # 儲存平台 JSON 檔案
    platform_json_path = output_series_folder.joinpath('rdx_cmb_pred_json.json')
    with open(platform_json_path, 'w') as f:
        f.write(platform_json.model_dump_json())
    #

    print("Processing complete!",platform_json_path)


if __name__ == '__main__':
    main()

