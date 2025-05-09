#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import asyncio
import aiofiles
import warnings
import pyorthanc

warnings.filterwarnings("ignore")  # 忽略警告输出
import os
import argparse
from pyorthanc import Orthanc,AsyncOrthanc
from dotenv import load_dotenv
load_dotenv()


async def upload_dicom_file(client, file_path):
    async with aiofiles.open(file_path,mode='rb') as f:
        content = await f.read()
        return await client.post_instances(content)


async def main():
    parser = argparse.ArgumentParser(description="處理DICOM-SEG檔案至Orthanc")
    parser.add_argument('--Input', type=str, nargs='+',
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/10516407_20231215_MR_21210200091/Pred_WMH_A1.dcm',
                        help='DICOM-SEG檔案')
    args = parser.parse_args()
    # upload_dicom_seg()
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")

    if isinstance(args.Input, str):
        ## Or with authentication:
        # client = Orthanc('http://localhost:8042', username='orthanc', password='orthanc')
        ## Connect to Orthanc server
        client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL)
        result = pyorthanc.upload(client, args.Input)
        print('result', result)

    if isinstance(args.Input, list):
        input_list = args.Input
        client = AsyncOrthanc(UPLOAD_DATA_DICOM_SEG_URL)
        tasks = [upload_dicom_file(client, file_path) for file_path in input_list]
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        print('result', results)


# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    asyncio.run(main())