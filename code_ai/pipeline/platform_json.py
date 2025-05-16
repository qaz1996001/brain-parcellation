#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import os
import warnings
from typing import List

import aiohttp
import orjson
from pydicom import FileDataset, dcmread
warnings.filterwarnings("ignore")  # 忽略警告输出
import argparse
# from code_ai import load_dotenv
# load_dotenv()
from pydantic import BaseModel,PositiveInt

# // 目前 Orthanc 自動同步機制的 group 應為 44
GROUP_ID = os.getenv("GROUP_ID",44)


class InstanceRequest(BaseModel):
    # dicom id
    sop_instance_uid :str
    # 應與 ImageOrientationPatient 和 ImagePositionPatient 有關
    projection       :str


class SeriesRequest(BaseModel):
    # dicom
    series_instance_uid :str
    instance : List[InstanceRequest]
    pass


class SortedRequest(BaseModel):
    study_instance_uid: str
    series            : List[SeriesRequest]
    pass



class StudyRequest(BaseModel):
    group_id           :int = GROUP_ID
    # "2017-12-25"
    study_date         :str
    # M F
    gender             :str
    # 99 56
    age                :int
    # Brain MRI
    study_name         :str
    # Mock
    patient_name       :str
    # 表幾顆腫瘤， 1.... 99
    aneurysm_lession   :PositiveInt
    # 表示 AI 訓練結果 , 1:成功, 2:進行中, 3:失敗
    aneurysm_status    :PositiveInt = 1
    resolution_x       :PositiveInt = 256
    resolution_y       :PositiveInt = 256
    study_instance_uid :str
    patient_id         :str


class MaskInstanceRequest(BaseModel):
    mask_index          :PositiveInt = 1
    mask_name           :str
    # diameter,type,location,sub_location,checked 在第一次上傳後寫入 db , 之後可從前端介面手動更改
    diameter            :str
    type                :str = 'saccular'
    location            :str = 'M'
    sub_location        :str = '2'
    prob_max            :str
    # bool , 0 或 1
    checked             :str = '1'
    # bool , 0 或 1
    is_ai               :str = '1'
    # 目前僅 Aneu 的 TOF_MRA:1 , Pitch: 2 , Yaw: 3 , 未來有其他 Series 再加
    series_type         :str
    #
    series_instance_uid :str

    sop_instance_uid    :str

    main_seg_slice      :int
    is_main_seg         :int = 0


class MaskRequest(BaseModel):
    study_instance_uid : str
    group_id           : PositiveInt = GROUP_ID
    instances          : List[MaskInstanceRequest]
    pass


class AITeamRequest(BaseModel):
    study   : StudyRequest
    sorted_ : SortedRequest
    mask    : MaskRequest


def build_study_json(dcm_ds:FileDataset,
                     group_id:int ,
                     aneurysm_lession:int,
                     aneurysm_status:int):
    # (0010,0010)	Patient Name	GUO HSIOU HUA
    # (0010,1010)	Patient Age	067Y
    # (0008,1030)	Study Description	MRI , Stroke Wall (-C +C)
    # (0020,000D)	Study Instance UID	1.2.840.113820.7134846831.158.821002010079.8
    # (0020,000E)	Series Instance UID	1.2.826.0.1.3680043.8.498.10901135655776388363626277680993721296
    # (0008,0018)	SOP Instance UID	1.2.826.0.1.3680043.8.498.10767331470938258007113825379434378798
    # (0008,0020)	Study Date	20210201
    pass


if __name__ == '__main__':
    # asyncio.run(main())
    parser = argparse.ArgumentParser(description="處理 AI predict 檔案至 web server")
    parser.add_argument('--Input', type=str, nargs='+',
                        default=['/mnt/e/pipeline/sean/rename_nifti/12472275_20231031_MR_21209070029/Pred_CMB_A1.dcm'],
                        help='json檔案')
    args = parser.parse_args()
    with open(args.Input[0],'rb') as f:
        dcm_ds = dcmread(f)
        print(dcm_ds)

