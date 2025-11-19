#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import asyncio
import pathlib

import warnings

import pandas as pd
import pydicom

warnings.filterwarnings("ignore")  # 忽略警告输出
import os
import pyorthanc
from pyorthanc import Orthanc,AsyncOrthanc, Study, Series
from dotenv import load_dotenv
load_dotenv()




def get_orthanc_study_uid_series_uid(instance_path_str:str):
    instance_path = pathlib.Path(instance_path_str)
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    # E:\raw_dicom\ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30\10089413 GUO HSIOU HUA\21002010079 MRI Stroke Wall C C\MR 3D Ax SWAN\*.dcm
    with open(instance_path_str,mode='rb') as f:
        dicom_ds = pydicom.dcmread(f)

    client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300)
    study_uid = instance_path.parent.parent.parent.parent.name
    # (0020,000E)	Series Instance UID	1.2.840.113619.2.44.5554020.7707121.19025.1612063861.703
    series_sop_uid = dicom_ds[0x0020, 0x000E].value
    # series_description = " ".join(instance_path.parent.name.split(" ")[1:]).strip()
    study = Study(study_uid,client=client)
    series_filter = list(filter(lambda series: series.uid == series_sop_uid, study.series))
    if series_filter:
        return str(study_uid),str(series_filter[0].id_)
    else:
        return None


def get_orthanc_series_uid(study_uid: str,
                           series_dir_set: set):
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300)
    series_sop_uid_list = []
    for series_dir in series_dir_set:
        series_path_list = list(series_dir.rglob('*.dcm'))
        instance_path_str = series_path_list[0]
        with open(instance_path_str, mode='rb') as f:
            dicom_ds = pydicom.dcmread(f)
        series_sop_uid = dicom_ds[0x0020, 0x000E].value
        series_sop_uid_list.append(series_sop_uid)
    study = Study(study_uid, client=client)
    series_dict_list = list(map(lambda x:{'series_sop_uid':x.uid,
                                          'uid': x.id_,
                                          'description':x.description,},study.series))
    df  = pd.DataFrame(series_sop_uid_list,columns=['file_series_sop_uid'])
    df1 = pd.DataFrame(series_dict_list)
    df2 = pd.merge(df,df1,left_on='file_series_sop_uid',right_on='series_sop_uid')
    return df2


def main():
    study_path    = pathlib.Path('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30')

    dcm_path_list = sorted(study_path.rglob("*.dcm"))
    series_dir_set = set([dcm_path.parent for dcm_path in dcm_path_list])
    df = get_orthanc_series_uid(study_uid=study_path.name,series_dir_set=series_dir_set)
    print(df)
    #
    # raw_parent_set = set()
    # for dcm_path in dcm_path_list:
    #     raw_series_path_dir_str = str(os.path.dirname(dcm_path))
    #
    #     if raw_series_path_dir_str in raw_parent_set:
    #         pass
    #     else:
    #         raw_parent_set.add(raw_series_path_dir_str)
    #         print('dcm_path',dcm_path)
    #         dir_result = get_orthanc_study_uid_series_uid(str(dcm_path))
    #         print('dir_result',dir_result)


if __name__ == '__main__':
    print('10000')
    # main()

    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    print('UPLOAD_DATA_DICOM_SEG_URL', UPLOAD_DATA_DICOM_SEG_URL)
    client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300)
