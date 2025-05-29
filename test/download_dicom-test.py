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


async def download_dicom(id_, output_path):
    """
    下載DICOM archive並解壓縮文件到指定路徑

    Args:
        id_: 要下載的series ID
        output_path: 輸出目錄路徑
    """
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    async_client = AsyncOrthanc(UPLOAD_DATA_DICOM_SEG_URL,timeout=300.0)

    try:
        # 確保輸出目錄存在
        os.makedirs(output_path, exist_ok=True)
        result = await async_client.post_series_id_archive(id_)
        # 檢查結果是否為bytes類型
        if isinstance(result, bytes):
            # 使用zipfile從記憶體中解壓縮
            with zipfile.ZipFile(io.BytesIO(result)) as zip_ref:
                # 獲取所有檔案名稱列表
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                print(f"Extracting {total_files} files from series {id_} to {output_path}")

                # 創建異步任務列表以解壓每個文件
                extract_tasks = []

                for file_index, file_name in enumerate(file_list):
                    # 從zip中獲取文件內容
                    file_content = zip_ref.read(file_name)

                    # 設定輸出的文件路徑
                    output_file_path = os.path.join(output_path, os.path.basename(file_name))

                    # 創建異步寫入任務

                    task = asyncio.create_task(write_file(output_file_path, file_content))
                    extract_tasks.append(task)

                    # 每處理50個文件輸出一次進度
                    if (file_index + 1) % 50 == 0 or file_index == total_files - 1:
                        print(f"Progress: {file_index + 1}/{total_files} files processed")

                # 等待所有文件寫入完成
                if extract_tasks:
                    await asyncio.gather(*extract_tasks)

                print(f"Successfully extracted {total_files} DICOM files from series {id_} to {output_path}")
            return True
        else:
            print(f"Unexpected result type: {type(result)}")
            print(f"Result content: {result}")
            print(result.text)
            return False
    except Exception as e:
        print(f"Error processing series {id_}: {str(e)}")
        traceback.print_exc()
        return False


async def main():
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL)

    # (0008,0060)   Modality   SEG
    # (0008,103E)   Series Description Pred_WMH
    AccessionNumber = '21002010079'
    level = 'Study'
    query = {'AccessionNumber': AccessionNumber,
            }
    result: List[Union[Study, Resource]] = query_orthanc(client=client, level=level, query=query)
    print('result', result)

    if len(result) == 0:
        print(f"No results found for AccessionNumber: {AccessionNumber}")
        return

    # 篩選MR系列
    mr_series = list(filter(lambda x: x.modality=='MR', result[0].series))
    print(f"Found {len(mr_series)} MR series")
    print(mr_series)

    if len(mr_series) > 0:
        # 創建輸出目錄
        study_id = result[0].id_
        #output_directory = f"/data/4TB1/raw_dicom/{study_id}"
        output_directory = f"/data/4TB1/raw_dicom/{study_id}"
        os.makedirs(output_directory, exist_ok=True)  # 確保目錄存在

        # 選擇性：也可以單獨下載每個MR系列
        download_tasks = []
        for series in mr_series:
            series_id = series.id_
            series_output_path = os.path.join(output_directory, f"{series_id}")
            # 使用asyncio.create_task創建任務以並行下載
            if os.path.exists(series_output_path):
                continue
            task = asyncio.create_task(
                download_dicom(series_id, series_output_path)
            )
            download_tasks.append(task)
            # break

        # 等待所有下載任務完成
        if download_tasks:
            print(f"Starting download of {len(download_tasks)} series...")
            await asyncio.gather(*download_tasks)
            print("All downloads completed!")
    else:
        print("No MR series found in the study")
        return

# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    asyncio.run(main())  # 使用asyncio.run來運行異步main函數
