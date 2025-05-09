# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:44:59 2025

根據手邊資料製作多組dicom-seg影像
由於國庭的cornerstone有bug，不能讀取出mask id，所以必須一個mask一個dicom-seg
範例case包含3組aneurysm跟1組血管

@author: user
"""
import argparse
import os
import pathlib

import numpy as np
import pydicom

import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pydicom_seg
import SimpleITK as sitk
import matplotlib.colors as mcolors



example_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resource',  'SEG_20230210_160056_635_S3.dcm')


#針對5類設計影像顏色
model_name = 'CMB_AI'

labels_vessel = {
           1: {
               "SegmentLabel": "CMB",
               "color":"red"
               }
    }

#讀範例文件，其實就是抓出某些tag格式不想自己設定
dcm_example = pydicom.dcmread(example_file)


#會使用到的一些predict技巧
def data_translate(img, nii):
    img = np.swapaxes(img,0,1)
    img = np.flip(img,0)
    img = np.flip(img, -1)
    header = nii.header.copy() #抓出nii header 去算體積 
    pixdim = header['pixdim']  #可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)  
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

def data_translate_back(img, nii):
    header = nii.header.copy() #抓出nii header 去算體積 
    pixdim = header['pixdim']  #可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)      
    img = np.flip(img, -1)
    img = np.flip(img,0)
    img = np.swapaxes(img,1,0)
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

#nii改變影像後儲存，data => nib.load的輸出，new_img => 更改的影像
def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii




def get_dicom_seg_template(model, label_dict):
    # key is model output label
    # values is meta data
    unique_labels = list(label_dict.keys())
    
    segment_attributes = []
    for i, idx in enumerate(unique_labels):
        name = label_dict[idx]["SegmentLabel"]
        
        rgb_rate = mcolors.to_rgb(label_dict[idx]["color"])
        rgb = [int(y*255) for y in rgb_rate]
        

        segment_attribute = {
                "labelID": int(idx),
                "SegmentLabel": name,
                "SegmentAlgorithmType": "MANUAL",
                "SegmentAlgorithmName": "MONAILABEL",
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
        "ContentDescription": "MONAI Label - Image segmentation",
        "ClinicalTrialCoordinatingCenterName": "MONAI",
        "BodyPartExamined": "",
    }
    return template


def make_dicomseg(mask, path_dcms, dcm_example, model_name, labels):
    #先做dicom-seg空格式
    template_json = get_dicom_seg_template(model_name, labels)
    template = pydicom_seg.template.from_dcmqi_metainfo(template_json)

    #可以使用 編寫多類分割MultiClassWriter，它有一些可用的自定義選項。如果分割數據非常稀疏，這些選項可以大大減少生成的文件大小。
    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,  # Crop image slices to the minimum bounding box on
                                # x and y axes
        skip_empty_slices=True,  # Don't encode slices with only zeros
        skip_missing_segment=True,  # If a segment definition is missing in the
                                     # template, then raise an error instead of
                                     # skipping it.
    )
    
    #這邊把資料填入dicom-seg結構中
    y_l, x_l, z_l = mask.shape
    #先排序並生成一個原版的dicom-seg，沒任何AI標註True不適合用套件做dicom-seg
    #if np.max(mask) > 0:
    #要從外部判斷是否適合做出dicom-seg，而不是內部用if
    #為了說明從分割數據創建 DICOM-SEG，典型的圖像分析工作流程如下所示。首先，圖像作為            
    reader = sitk.ImageSeriesReader()
    dcms = sorted(reader.GetGDCMSeriesFileNames(path_dcms))
        
    #Stack the 2D slices to form a 3D array representing the volume
    slices = [pydicom.dcmread(dcm) for dcm in dcms]
        
    slice_dcm = []
    #需要依照真實切片位置來更新順序
    for (slice, dcm_slice) in zip(slices, dcms):
        # (0020,0037)	Image Orientation Patient	0.996629\0.0818798\0.0050909\-0.0820347\0.994121\0.0706688
        # (0020,0032)	Image Position Patient	-113.712\-154.663\-44.8992
        IOP = np.array(slice.get((0x0020, 0x0037)).value)
        IPP = np.array(slice.get((0x0020, 0x0032)).value)
        # IOP = np.array(slice.ImageOrientationPatient)
        # IPP = np.array(slice.ImagePositionPatient)
        normal = np.cross(IOP[0:3], IOP[3:])
        projection = np.dot(IPP, normal)
        slice_dcm += [{"d": projection, "dcm": dcm_slice}]
        
    #sort the slices，sort是由小到大，但頭頂是要由大到小，所以要反向
    #由於dicom-seg套件影像排序就是照reader.GetGDCMSeriesFileNames，所以不用再自己改頭到腳(原本腳到頭)
    slice_dcms = sorted(slice_dcm, key = lambda i: i['d'])
        
    new_dcms = [y['dcm'] for y in slice_dcms]
    new_dcms = new_dcms
    #print('new_dcms:', new_dcms)
        
    #接下來讀取第一張影像
    dcm_one = pydicom.dcmread(new_dcms[0], force=True) #dcm的切片
        
    #重建空影像給dicomSEG
    reader.SetFileNames(new_dcms)
    image = reader.Execute()
    #image_data = sitk.GetArrayFromImage(image)

    #將原本label的shape轉換成 z,y,x，把影像重新存入
    #mask = np.flip(mask, 0)
    #mask = np.flip(mask, -1) #製作dicom-seg時標註要腳到頭
    segmentation_data = mask.transpose(2, 0, 1).astype(np.uint8)
    segmentation_data = np.swapaxes(segmentation_data,1,2)
    segmentation_data = np.flip(segmentation_data, 1) #y軸
    segmentation_data = np.flip(segmentation_data, 2)
    #segmentation_data = np.flip(segmentation_data, 0)
    #print('segmentation_data.shape:', segmentation_data.shape)

    #由於pydicom_seg期望 aSimpleITK.Image作為作者的輸入，因此需要將分割數據與有關圖像網格的所有相關信息一起封裝。
    #這對於編碼片段和引用相應的源 DICOM 文件非常重要。
    segmentation = sitk.GetImageFromArray(segmentation_data)
    segmentation.CopyInformation(image)
    

    #最後，可以執行 DICOM-SEG 的實際創建。因此，需要加載源 DICOM 文件pydicom，但出於優化目的，可以跳過像素數據
    source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in new_dcms]
    dcm_seg_ = writer.write(segmentation, source_images)

    #增加一些標註需要的資訊，作法，輸入範例資訊，再把實際case的值填入
    dcm_seg_[0x10,0x0010].value = dcm_one[0x10,0x0010].value #(0010,0010) Patient's Name <= (0010,0020) Patient ID
    dcm_seg_[0x20,0x0011].value = dcm_one[0x20,0x0011].value # (0020,0011) Series Number [3]
    #dcm_seg[0x62,0x0002].value = dcm_example[0x62,0x0002].value #Segment Sequence Attribute
    dcm_seg_[0x5200,0x9229].value = dcm_example[0x5200,0x9229].value
    dcm_seg_[0x5200,0x9229][0][0x20,0x9116][0][0x20,0x0037].value = dcm_one[0x20,0x0037].value #(0020,0037) Image Orientation (Patient)
    dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x18,0x0050].value = dcm_one[0x18,0x0050].value #(0018,0050) Slice Thickness 
    dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x18,0x0088].value = dcm_one[0x18,0x0088].value #(0018,0088) Spacing Between Slices
    dcm_seg_[0x5200,0x9229][0][0x28,0x9110][0][0x28,0x0030].value = dcm_one[0x28,0x0030].value #(0028,0030) Pixel Spacing    
    
    return dcm_seg_



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=str, default='10516407_20231215_MR_21210200091',
                        help='目前執行的case的patient_id or study id')
    parser.add_argument('--InputsDicom', type=str, default='/mnt/e/rename_dicom_202505051/10516407_20231215_MR_21210200091/SWAN',
                        help='用於輸出結果的資料夾')

    parser.add_argument('--InputsNifti', type=str,
                        default= '/mnt/e/rename_nifti_202505051/10516407_20231215_MR_21210200091/Pred_CMB.nii.gz',
                        help='用於輸入的檔案')

    parser.add_argument('--OutputDicomSegFolder', type=str, default='/mnt/e/dicom_seg_202505051/',
                        help='用於輸出結果的資料夾')

    args = parser.parse_args()
    ID          = args.ID
    path_dcms   = pathlib.Path(args.InputsDicom)
    path_nii    = pathlib.Path(args.InputsNifti)
    path_dcmseg = pathlib.Path(args.OutputDicomSegFolder)

    if path_dcmseg.is_dir():
        path_dcmseg.mkdir(parents=True, exist_ok=True)
    else:
        path_dcmseg.parent.mkdir(parents=True, exist_ok=True)

    series = path_nii.name.split('.')[0]
    Pred_nii = nib.load(path_nii)
    Pred = np.array(Pred_nii.dataobj)

    series_folder = path_dcmseg.joinpath(f'{ID}')
    series_folder.mkdir(exist_ok=True, parents=True)
    for i in range(int(np.max(Pred))):
        new_Pred = np.zeros((Pred.shape))
        new_Pred[Pred==i+1] = 1

        labels_ane = {}
        labels_ane[1] = {'SegmentLabel': 'A' + str(i+1), 'color': 'red'}

        #底下做成dicom-seg
        if np.sum(new_Pred):
            dcm_seg = make_dicomseg(new_Pred.astype('uint8'), str(path_dcms), dcm_example, model_name, labels_ane)
            dcm_seg_path = series_folder.joinpath('{}_{}.dcm'.format(series,labels_ane[1]['SegmentLabel']))
            dcm_seg.save_as(dcm_seg_path)