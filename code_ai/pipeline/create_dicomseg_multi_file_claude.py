# -*- coding: utf-8 -*-
"""

根據手邊資料製作多組dicom-seg影像
由於國庭的cornerstone有bug，不能讀取出mask id，所以必須一個mask一個dicom-seg
範例case包含為CMB

@author: sean
"""

import argparse
import os
import pathlib
from typing import Dict, List, Any

import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
import matplotlib.colors as mcolors
import pydicom_seg
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir

# 載入範例文件只需執行一次，改為在函數外
# EXAMPLE_FILE = 'SEG_20230210_160056_635_S3.dcm'

EXAMPLE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'resource', 'SEG_20230210_160056_635_S3.dcm')
DCM_EXAMPLE = pydicom.dcmread(EXAMPLE_FILE)

MODEL_NAME = 'CMB_AI'


def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin


def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)

    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)


def get_dicom_seg_template(model: str, label_dict: Dict) -> Dict:
    """創建DICOM-SEG模板"""
    unique_labels = list(label_dict.keys())

    segment_attributes = []
    for idx in unique_labels:
        name = label_dict[idx]["SegmentLabel"]
        rgb_rate = mcolors.to_rgb(label_dict[idx]["color"])
        rgb = [int(y * 255) for y in rgb_rate]

        segment_attribute = {
            "labelID": int(idx),
            "SegmentLabel": name,
            "SegmentAlgorithmType": "MANUAL",
            "SegmentAlgorithmName": "SHH",
            "SegmentedPropertyCategoryCodeSequence": {
                "CodeValue": "M-01000",
                "CodingSchemeDesignator": "SRT",
                "CodeMeaning": "Morphologically Altered Structure",
            },
            "SegmentedPropertyTypeCodeSequence": {
                "CodeValue": "M-35300",
                "CodingSchemeDesignator": "SRT",
                "CodeMeaning": "Embolus",
            },
            "recommendedDisplayRGBValue": rgb,
        }
        segment_attributes.append(segment_attribute)

    template = {
        "ContentCreatorName": "Reader1",
        "ClinicalTrialSeriesID": "Session1",
        "ClinicalTrialTimePointID": "1",
        "SeriesDescription": model,
        "SeriesNumber": "300",
        "InstanceNumber": "1",
        "segmentAttributes": [segment_attributes],
        "ContentLabel": "SEGMENTATION",
        "ContentDescription": "SHH",
        "ClinicalTrialCoordinatingCenterName": "SHH",
        "BodyPartExamined": "",
    }
    return template


def load_and_sort_dicom_files(path_dcms: str) -> tuple[
    list[Any], Any, FileDataset | DicomDir, list[FileDataset | DicomDir]]:
    """載入並排序DICOM檔案，只需執行一次"""
    # 讀取DICOM文件
    reader = sitk.ImageSeriesReader()
    print('path_dcms',type(path_dcms),path_dcms)
    dcms = sorted(reader.GetGDCMSeriesFileNames(path_dcms))

    # 讀取所有切片
    slices = [pydicom.dcmread(dcm) for dcm in dcms]

    # 依照切片位置排序
    slice_dcm = []
    for (slice_data, dcm_slice) in zip(slices, dcms):
        IOP = np.array(slice_data.get((0x0020, 0x0037)).value)
        IPP = np.array(slice_data.get((0x0020, 0x0032)).value)
        normal = np.cross(IOP[0:3], IOP[3:])
        projection = np.dot(IPP, normal)
        slice_dcm.append({"d": projection, "dcm": dcm_slice})

    # 排序切片
    slice_dcms = sorted(slice_dcm, key=lambda i: i['d'])
    sorted_dcms = [y['dcm'] for y in slice_dcms]

    # 讀取影像資料
    reader.SetFileNames(sorted_dcms)
    image = reader.Execute()

    # 讀取第一張DICOM影像以獲取標頭資訊
    first_dcm = pydicom.dcmread(sorted_dcms[0], force=True)

    # 預載所有DICOM檔案（但不載入像素資料以節省記憶體）
    source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in sorted_dcms]

    return sorted_dcms, image, first_dcm, source_images


def transform_mask_for_dicom_seg(mask: np.ndarray) -> np.ndarray:
    """將遮罩轉換為DICOM-SEG所需的格式"""
    # 轉換格式：(y,x,z) -> (z,x,y)
    segmentation_data = mask.transpose(2, 0, 1).astype(np.uint8)
    # 轉換格式：(z,x,y) -> (z,y,x)
    segmentation_data = np.swapaxes(segmentation_data, 1, 2)
    # 翻轉y軸和x軸以符合DICOM座標系統
    segmentation_data = np.flip(segmentation_data, 1)
    segmentation_data = np.flip(segmentation_data, 2)

    return segmentation_data


def make_dicomseg_file(
        mask: np.ndarray,
        image: sitk.Image,
        first_dcm: pydicom.FileDataset,
        source_images: List[pydicom.FileDataset],
        template_json: Dict
) -> pydicom.FileDataset:
    """製作DICOM-SEG檔案"""
    # 創建模板
    template = pydicom_seg.template.from_dcmqi_metainfo(template_json)

    # 設定寫入器
    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,
        skip_empty_slices=True,
        skip_missing_segment=True,
    )

    # 轉換遮罩格式
    #segmentation_data = transform_mask_for_dicom_seg(mask)
    segmentation_data = mask

    # 創建SimpleITK影像
    segmentation = sitk.GetImageFromArray(segmentation_data)
    segmentation.CopyInformation(image)

    # 產生DICOM-SEG檔案
    dcm_seg = writer.write(segmentation, source_images)

    # 從示範檔案和第一張DICOM影像中複製相關資訊
    dcm_seg[0x10, 0x0010].value = first_dcm[0x10, 0x0010].value  # Patient's Name
    dcm_seg[0x20, 0x0011].value = first_dcm[0x20, 0x0011].value  # Series Number

    # 複製更多元數據
    dcm_seg[0x5200, 0x9229].value = DCM_EXAMPLE[0x5200, 0x9229].value
    dcm_seg[0x5200, 0x9229][0][0x20, 0x9116][0][0x20, 0x0037].value = first_dcm[0x20, 0x0037].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x18, 0x0050].value = first_dcm[0x18, 0x0050].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x18, 0x0088].value = first_dcm[0x18, 0x0088].value
    dcm_seg[0x5200, 0x9229][0][0x28, 0x9110][0][0x28, 0x0030].value = first_dcm[0x28, 0x0030].value

    return dcm_seg


def process_prediction_mask(
        pred_data: np.ndarray,
        path_dcms: str,
        series_name: str,
        output_folder: pathlib.Path
) -> None:
    """處理預測遮罩並生成DICOM-SEG文件"""
    # 只需載入一次DICOM檔案
    sorted_dcms, image, first_dcm, source_images = load_and_sort_dicom_files(path_dcms)

    # 處理每個分割區域
    for i in range(int(np.max(pred_data))):
        # 創建單一區域的遮罩
        mask = np.zeros_like(pred_data)
        mask[pred_data == i + 1] = 1

        # 如果遮罩中有值，才創建DICOM-SEG
        if np.sum(mask) > 0:
            # 創建標籤字典
            label_dict = {1: {'SegmentLabel': f'A{i + 1}', 'color': 'red'}}

            # 創建模板
            template_json = get_dicom_seg_template(series_name, label_dict)

            # 產生DICOM-SEG
            dcm_seg = make_dicomseg_file(
                mask.astype('uint8'),
                image,
                first_dcm,
                source_images,
                template_json
            )

            # 保存DICOM-SEG
            dcm_seg_filename = f'{series_name}_{label_dict[1]["SegmentLabel"]}.dcm'
            dcm_seg_path = output_folder.joinpath(dcm_seg_filename)
            dcm_seg.save_as(dcm_seg_path)
            print(f"Saved: {dcm_seg_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="處理NIFTI檔案並創建DICOM-SEG檔案")
    parser.add_argument('--ID', type=str, default='10516407_20231215_MR_21210200091',
                        help='目前執行的case的patient_id or study id')
    # parser.add_argument('--InputsDicom', type=str,
    #                     default='/mnt/e/rename_dicom_202505051/10516407_20231215_MR_21210200091/SWAN',
    #                     help='用於輸出結果的資料夾')
    # parser.add_argument('--InputsNifti', type=str,
    #                     default='/mnt/e/rename_nifti_202505051/10516407_20231215_MR_21210200091/Pred_CMB.nii.gz',
    #                     help='用於輸入的檔案')
    # parser.add_argument('--OutputDicomSegFolder', type=str,
    #                     default='/mnt/e/dicom_seg_202505051/',
    #                     help='用於輸出結果的資料夾')
    parser.add_argument('--InputsDicom', type=str,
                        default='/mnt/e/rename_dicom_202505051/10516407_20231215_MR_21210200091/T2FLAIR_AXI',
                        help='用於輸出結果的資料夾')
    parser.add_argument('--InputsNifti', type=str,
                        default='/mnt/e/rename_nifti_202505051/10516407_20231215_MR_21210200091/Pred_WMH.nii.gz',
                        help='用於輸入的檔案')
    parser.add_argument('--OutputDicomSegFolder', type=str,
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/',
                        help='用於輸出結果的資料夾')


    args = parser.parse_args()
    ID = args.ID
    path_dcms = pathlib.Path(args.InputsDicom)
    path_nii = pathlib.Path(args.InputsNifti)
    path_dcmseg = pathlib.Path(args.OutputDicomSegFolder)

    # 創建輸出目錄
    if path_dcmseg.is_dir():
        path_dcmseg.mkdir(parents=True, exist_ok=True)
    else:
        path_dcmseg.parent.mkdir(parents=True, exist_ok=True)

    # 取得系列名稱
    series = path_nii.name.split('.')[0]

    # 載入預測數據
    pred_nii = nib.load(path_nii)
    pred_data = np.array(pred_nii.dataobj)

    pred_nii_obj_axcodes = tuple(nib.aff2axcodes(pred_nii.affine))
    new_nifti_array = do_reorientation(pred_data, pred_nii_obj_axcodes, ('S', 'P', 'L'))

    # 創建輸出子目錄
    series_folder = path_dcmseg.joinpath(f'{ID}')
    series_folder.mkdir(exist_ok=True, parents=True)
    # 處理預測遮罩
    process_prediction_mask(new_nifti_array, str(path_dcms), series, series_folder)
    print("Processing complete!")


if __name__ == '__main__':
    main()
