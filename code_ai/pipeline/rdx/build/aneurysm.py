import pathlib
from typing import Dict, Any, Union, List

import numpy as np
import pandas as pd
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir

from code_ai.pipeline.dicomseg import utils

from code_ai.pipeline.rdx.schema import AneurysmDetectionItem, AneurysmDetectionResponse
from .base import PredictionBaseBuilder


class AneurysmDetectionBuilder(PredictionBaseBuilder[AneurysmDetectionItem, AneurysmDetectionResponse]):
    """
    動脈瘤檢測結果 Builder
    """

    model_class = AneurysmDetectionResponse

    angle_key_mapping = {
        'MIP_Pitch':'pitch_angle',
        'MIP_Yaw':'yaw_angle'
    }

    @staticmethod
    def get_excel_to_pred_json(excel_file_path: str,
                               intput_list: List[Dict[str, Any]]):
        """
        intput_list:Dict[str,str] -> {"series_name": "MIP_Pitch",
             #                        "pred_nii_path": "*.nii.gz"}
        """
        pred_json_list = []
        df = pd.read_excel(excel_file_path)
        aneurysm_number_count = df['Aneurysm_Number'].iloc()[0]
        for intput_dict in intput_list:
            series_name_pred = []
            for aneurysm_number in range(1, aneurysm_number_count + 1):
                pred = utils.get_array_to_dcm_axcodes(intput_dict['pred_nii_path'])
                have_labels = np.where(pred == aneurysm_number)[0]
                sub_location = df[f"{aneurysm_number}_SubLocation"].iloc()[0]
                segment_attribute = {
                    "mask_index": str(aneurysm_number),
                    "mask_name": 'A' + str(aneurysm_number),
                    "diameter": str(round(df[f"{aneurysm_number}_size"].iloc()[0], 1)),
                    "type": 'saccular',
                    "location": str(df[f"{aneurysm_number}_Location"].iloc()[0]),
                    "sub_location": sub_location if pd.notna(sub_location) else "",
                    "probability": round(df[f"{aneurysm_number}_Prob_max"].iloc()[0], 2),
                    "main_seg_slice": int(pred.shape[0]) - int(np.median(have_labels) + 1),
                }
                series_name_pred.append(segment_attribute)
            pred_json_list.append({"series_name": intput_dict['series_name'],
                                   "data": series_name_pred
                                   })
        return pred_json_list


        return pred_json_list

    @staticmethod
    def use_create_dicom_seg_file(path_nii: pathlib.Path,
                                  series_name: str,
                                  output_folder: pathlib.Path,
                                  image: Any,
                                  first_dcm: FileDataset | DicomDir,
                                  source_images: List[FileDataset | DicomDir], ):
        # Load prediction data from NIfTI file
        if series_name == "MRA_BRAIN":
            new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii.joinpath(f'Pred.nii.gz'))
        else:
            new_nifti_array = utils.get_array_to_dcm_axcodes(path_nii.joinpath(f'{series_name}_pred.nii.gz'))
        pred_data_unique = np.unique(new_nifti_array)
        if len(pred_data_unique) < 1:
            return None
        else:
            pred_data_unique = pred_data_unique[1:]  # Exclude background value (0)
        result_list = utils.create_dicom_seg_file(pred_data_unique,
                                                  new_nifti_array,
                                                  series_name,
                                                  output_folder,
                                                  image,
                                                  first_dcm,
                                                  source_images)
        return result_list

    @classmethod
    def merge_pitch_yaw_angle(cls, pred_json_list:List[Dict[str, Any]],series_name:str = 'MRA_BRAIN') -> Dict[str, Any]:

        mra_brain_json = next(filter(lambda x: x['series_name'] == series_name, pred_json_list))
        pitch_yaw_list = list(filter(lambda x: x['series_name'] != series_name, pred_json_list))

        angle_list     = [ list(map(lambda x: {'key':pred_json['series_name'],
                                               x['mask_name']:x['main_seg_slice'] * 3 -3 }, pred_json['data']))for pred_json in pitch_yaw_list]
        mra_brain_json['data'] = [{**brain_json_data[0],
                                   **{cls.angle_key_mapping[xx['key']]: xx[brain_json_data[0]['mask_name']]
                                      for xx in brain_json_data[1:]}} for brain_json_data in zip(mra_brain_json['data'], *angle_list)]

        return mra_brain_json

    def build_detection(self,
                        source_image: Union[FileDataset, DicomDir],
                        prediction_result: Dict[str, Any],
                        *args, **kwargs) -> AneurysmDetectionItem:
        """
        構建單個動脈瘤檢測項目

        Args:
            source_image: DICOM 影像
            prediction_result: 包含以下鍵值的字典:
                - type: 動脈瘤類型
                - location: 位置
                - diameter: 直徑
                - main_seg_slice: 主要分割切片
                - probability: 機率
                - pitch_angle: 俯仰角
                - yaw_angle: 偏航角
                - mask_index: mask 索引
                - sub_location: 子位置

        Returns:
            AneurysmDetectionItem
        """
        # 從 DICOM 提取必要資訊
        series_instance_uid = source_image.get((0x0020, 0x000E)).value
        sop_instance_uid = source_image.get((0x0008, 0x0018)).value

        # 構建 detection 字典
        detection_dict = {
            'series_instance_uid': series_instance_uid,
            'sop_instance_uid': sop_instance_uid,
            'label': prediction_result.get('mask_name', 'aneurysm'),
            'type': prediction_result['type'],
            'location': prediction_result['location'],
            'diameter': prediction_result['diameter'],
            'main_seg_slice': prediction_result['main_seg_slice'],
            'probability': prediction_result['probability'],
            'pitch_angle': prediction_result.get('pitch_angle', 0),
            'yaw_angle': prediction_result.get('yaw_angle', 0),
            'mask_index': prediction_result['mask_index'],
            'sub_location': prediction_result.get('sub_location', ""),
        }

        return AneurysmDetectionItem.model_validate(detection_dict)

    @classmethod
    def execute_rdx_platform_json(cls,
                                  _id: int, path_root: pathlib.Path,
                                  model_id: str = '5d7b5e3a-9c1f-4a2b-8d6e-3f9a1c2b4e5f'):
        """
        執行 RDX 平台 JSON 生成流程，處理動脈瘤檢測的 DICOM 和預測結果

        Args:
            _id: 患者或案例 ID
            path_root: 資料根目錄路徑
            model_id: 模型識別碼，預設為動脈瘤檢測模型 ID
        """
        import pydicom

        # 定義所有必要的資料夾路徑
        path_dict = dict(
            path_dcms=path_root.joinpath("Dicom"),  # DICOM 原始檔案目錄
            path_nii=path_root.joinpath("Image_nii"),  # NIfTI 格式影像目錄
            path_reslice_nii=path_root.joinpath("Image_reslice"),  # 重切影像目錄
            path_excel=path_root.joinpath("excel"),  # Excel 結果目錄
            path_dcmseg=path_root.joinpath("Dicom", "Dicom-Seg")  # DICOM-SEG 輸出目錄
        )

        # 初始化動脈瘤檢測 JSON 建構器
        platform_json_builder = cls()

        # 定義三個序列名稱：腦部 MRA、俯仰角 MIP、偏航角 MIP
        series_name_list = ['MRA_BRAIN', 'MIP_Pitch', 'MIP_Yaw']

        # 讀取每個序列的 DICOM 資料夾路徑列表
        path_dcms_folder_path_list = [path_dict['path_dcms'].joinpath(x) for x in series_name_list]

        # 建立預測 NIfTI 檔案路徑列表
        # MRA_BRAIN 使用 Image_nii 目錄，其他使用 Image_reslice 目錄
        pred_nii_path_list = [{"series_name": x,
                               "pred_nii_path": str(path_dict['path_nii'].joinpath("Pred.nii.gz"))
                               if x == 'MRA_BRAIN'
                               else str(path_dict['path_reslice_nii'].joinpath("{}_pred.nii.gz".format(x)))
                               }
                              for x in series_name_list]

        # 載入並排序每個序列的 DICOM 檔案
        dicom_data_list = [utils.load_and_sort_dicom_files(x) for x in path_dcms_folder_path_list]
        series_first_dcm_data_list = [x[2] for x in dicom_data_list]

        # 從 Excel 檔案讀取動脈瘤檢測結果
        excel_file_path = str(path_dict['path_excel'].joinpath('Aneurysm_Pred_list.xlsx'))
        pred_json_list = cls.get_excel_to_pred_json(excel_file_path, pred_nii_path_list)

        # 為每個序列創建 DICOM-SEG 檔案
        pred_result_list = [{"series_name": x[0],
                             "data": cls.use_create_dicom_seg_file(
                                 path_dict['path_nii'] if x[0] == 'MRA_BRAIN' else path_dict['path_reslice_nii'],
                                 x[0],
                                 path_dict['path_dcmseg'],
                                 *x[1][1:]  # 解包 DICOM 資料（跳過第一個元素）
                             )}
                            for x in zip(series_name_list, dicom_data_list)]

        # 合併 pitch 和 yaw 角度資訊到 MRA_BRAIN 序列
        merged_pred_json = cls.merge_pitch_yaw_angle(pred_json_list, series_name='MRA_BRAIN')

        # 讀取所有生成的 DICOM-SEG 檔案
        dcm_seg_path_list = [list(map(lambda xx: pydicom.read_file(xx['dcm_seg_path']), x['data']))
                             for x in pred_result_list if x['data'] is not None]

        # 使用建構器模式組裝最終的平台 JSON
        platform_json = (platform_json_builder
                         .set_patient_info(series_first_dcm_data_list)  # 設定患者資訊
                         .set_model_id(model_id)  # 設定模型 ID
                         .set_detections(dcm_seg_path_list[0] if dcm_seg_path_list else [], merged_pred_json['data'])  # 設定檢測結果
                         .build()  # 建構最終 JSON
                                  )

        # 確保輸出目錄存在
        output_series_folder = path_root
        if not output_series_folder.is_dir():
            output_series_folder.mkdir(exist_ok=True, parents=True)

        # 儲存平台 JSON 檔案
        platform_json_path = output_series_folder.joinpath('rdx_aneurysm_json.json')
        with open(platform_json_path, 'w') as f:
            f.write(platform_json.model_dump_json())

        return platform_json

