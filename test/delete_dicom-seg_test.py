#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import os
import warnings
warnings.filterwarnings("ignore")  # 忽略警告输出
from pyorthanc import Orthanc, find_instances,AsyncOrthanc
from code_ai import load_dotenv
load_dotenv()


def del_synthseg_seg():
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL)
    # (0008,0060)	Modality	SEG
    # (0008,103E)	Series Description	Pred_WMH
    instances = find_instances(
        client,
        query={'Modality': 'SEG',
               'SeriesDescription': 'synthseg_*',
               },
    )
    print('instances:', instances)
    for instance in instances:
        client.delete_instances_id(instance.id_)

def del_pred_seg():
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL)
    instances = find_instances(
        client,
        query={'Modality': 'SEG',
               'SeriesDescription': 'Pred_*',
               },
    )
    for instance in instances:
        client.delete_instances_id(instance.id_)
    print('instances:', instances)

def main():
    # delete dicom seg
    del_synthseg_seg()
    del_pred_seg()

    # instances = find_instances(
    #     client,
    #     query={'Modality': 'SEG',
    #            'SeriesDescription': 'Aneurysm_AI',
    #            },
    # )
    # print('instances:', instances)
    # for instance in instances:
    #     client.delete_instances_id(instance.id_)
# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    main()