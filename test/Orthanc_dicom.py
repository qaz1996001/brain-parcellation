#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import asyncio
import glob
import pathlib

import aiofiles
import warnings
import pyorthanc

from code_ai.utils.inference import build_analysis, build_inference_cmd

warnings.filterwarnings("ignore")  # 忽略警告输出
import os
import pyorthanc
from pyorthanc import Orthanc  ,find_instances,Instance
from dotenv import load_dotenv
load_dotenv()


def main():
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL)
    instances = find_instances(
        client=client,
        query={'SeriesDescription': 'synthseg_*'},
    )
    for instance in instances:
        # print(instance.uid)
        print(instance.id_)
        print(client.delete_instances_id(instance.id_))
    instances = find_instances(
        client=client,
        query={'SeriesDescription': 'Pred_*'},
    )
    for instance in instances:
        # print(instance.uid)
        print(instance.id_)
        print(client.delete_instances_id(instance.id_))


if __name__ == '__main__':
    #
    #
    study_path       =  pathlib.Path('/mnt/e/rename_nifti_20250509/10089413_20210201_MR_21002010079')
    nifti_study_path = pathlib.Path('/mnt/e/rename_nifti_20250509/10089413_20210201_MR_21002010079')
    dicom_study_path = pathlib.Path('/mnt/e/rename_dicom_20250509/10089413_20210201_MR_21002010079')
    inference_cmd    = build_inference_cmd(nifti_study_path,dicom_study_path)
    print(inference_cmd)
    # main()