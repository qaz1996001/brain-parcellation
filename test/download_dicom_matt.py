"""
@author: sean Ho
"""
import io
import os
import pathlib
import traceback
import asyncio
from dataclasses import dataclass
import aiofiles
from pyorthanc import AsyncOrthanc, Instance
from code_ai import load_dotenv

load_dotenv()
semaphore = asyncio.Semaphore(512)


@dataclass
class DownloadDicom:
    async_client     :AsyncOrthanc
    output_directory :pathlib.Path
    study_uid        :str
    series_uid       :str
    instances_uid    :str



async def write_file(file_path, content):
    async with semaphore:
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

# async def download_dicom(id_, output_path):
#     UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
#     async_client = AsyncOrthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300.0)
#     # 確保輸出目錄存在
#     os.makedirs(output_path, exist_ok=True)
#     try:
#         result = await async_client.post_studies_id_archive(id_)
#
#         if isinstance(result, bytes):
#             # 使用zipfile從記憶體中解壓縮
#             with zipfile.ZipFile(io.BytesIO(result)) as zip_ref:
#                 zip_ref.extractall(output_path)
#             return True
#         else:
#             print(f"Result content: {result}")
#             return False
#     except Exception as e:
#         print(f"Error processing series {id_}: {str(e)}")
#         traceback.print_exc()
#         return False


async def download_dicom(download_dicom_data :DownloadDicom):
    output_path = download_dicom_data.output_directory.joinpath(download_dicom_data.study_uid,
                                                                download_dicom_data.series_uid,
                                                                f'{download_dicom_data.instances_uid}.dcm')
    async_client = download_dicom_data.async_client
    # 確保輸出目錄存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print('output_path',output_path)
    try:
        print('download_dicom_data.instances_uid,',download_dicom_data.instances_uid,)
        async with semaphore:
            study = Instance(study_uid, client=client)
            Instance()
        print('instances_response length:', len(instances_response))
        await write_file(output_path, instances_response)
        return True
    except Exception as e:
        print(f"Error processing series {download_dicom_data.instances_uid}: {str(e)}")
        traceback.print_exc()
        return False


async def main():
    download_tasks = []
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    async_client = AsyncOrthanc(UPLOAD_DATA_DICOM_SEG_URL, timeout=300.0)
    study_uid_list = [
        "dc7c15d1-881e24ab-7fbee4e7-f2eea96b-cd6fb500",
        # "f5a17cf9-6315d382-ce1dbf05-06e99a23-b810186e",
        # "ee3cd0a8-b0cd993d-155d2b72-78cb19cd-8abe296d",
        # "887f052d-9c31bf17-fc2b65b4-f55e6b29-d1b368bd"
    ]
    async_client
    output_directory = pathlib.Path("/mnt/e/test/pipeline/raw_dicom/")
    for study_uid in study_uid_list:
        studies = await async_client.get_studies_id(study_uid)
        series_uid_list = studies['Series']
        for series_uid in series_uid_list:
            series = await async_client.get_series_id(series_uid)
            instances_uid_list = series['Instances']
            async_client.post_instances_id_export()
            for instances_uid in instances_uid_list:
                download_dicom_data = DownloadDicom(async_client = async_client,
                                                    output_directory = output_directory,
                                                    study_uid = study_uid,
                                                    series_uid = series_uid,
                                                    instances_uid = instances_uid)
                task = asyncio.create_task(
                    download_dicom(download_dicom_data)
                )
                download_tasks.append(task)
                break
            break
        break
    print('download_tasks',download_tasks)

    results = await asyncio.gather(*download_tasks, return_exceptions=True)
    print('main results',results)
    # successful_count = sum(1 for result in results if result is True)
    # print(
    #     f"Batch processing completed! {successful_count}/{len(results)} studies processed successfully."
    # )


# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == "__main__":
    print("10000")
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
