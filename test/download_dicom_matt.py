"""
@author: sean Ho
"""
import io
import os
import traceback
import warnings
import asyncio
import zipfile
import aiofiles
from typing import List, Optional, Union

warnings.filterwarnings("ignore")  # 忽略警告输出
from pyorthanc import AsyncOrthanc, Orthanc, find_studies, query_orthanc, Study
from pyorthanc._resources import Resource
from code_ai import load_dotenv

load_dotenv()
semaphore = asyncio.Semaphore(512)


async def write_file(file_path, content):
    async with semaphore:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)


async def download_dicom_study(id_, output_path):
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    async_client = AsyncOrthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300.0)
    # 確保輸出目錄存在
    os.makedirs(output_path, exist_ok=True)
    try:
        result = await async_client.post_studies_id_archive(id_)

        if isinstance(result, bytes):
            # 使用zipfile從記憶體中解壓縮
            with zipfile.ZipFile(io.BytesIO(result)) as zip_ref:
                zip_ref.extractall(output_path)
            return True
        else:
            print(f"Result content: {result}")
            return False
    except Exception as e:
        print(f"Error processing series {id_}: {str(e)}")
        traceback.print_exc()
        return False


async def main():
    download_tasks = []
    study_uid = "55ba9d47-0982e704-bdb2bea6-95bcb9e9-9e49b3e4"
    output_directory = f"/mnt/e/raw_dicom/{study_uid}"
    task = asyncio.create_task(
        download_dicom_study(id_=study_uid, output_path=output_directory)
    )
    download_tasks.append(task)

    results = await asyncio.gather(*download_tasks, return_exceptions=True)
    successful_count = sum(1 for result in results if result is True)
    print(f"Batch processing completed! {successful_count}/{len(results)} studies processed successfully.")

# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    print('10000')
    asyncio.run(main())  # 使用asyncio.run來運行異步main函數
    # file_ = '/mnt/c/Users/user/Downloads/55ba9d47-0982e704-bdb2bea6-95bcb9e9-9e49b3e4.zip'
    # with zipfile.ZipFile(file_) as zip_ref:
    #     # 獲取所有檔案名稱列表
    #     file_list = zip_ref.namelist()
    #     total_files = len(file_list)
    #     zip_ref.extractall()
    #
    #     print('namelist', file_list)
    #     print('file_list', zip_ref.filelist)
