"""
@author: sean Ho
"""
import pathlib
import traceback
import asyncio
from dataclasses import dataclass
import aiofiles
from pyorthanc import AsyncOrthanc
from code_ai import load_dotenv

load_dotenv()
semaphore = asyncio.Semaphore(256)


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


async def download_dicom(download_dicom_data :DownloadDicom):
    output_path = download_dicom_data.output_directory.joinpath(download_dicom_data.study_uid,
                                                                download_dicom_data.series_uid,
                                                                f'{download_dicom_data.instances_uid}.dcm')
    if output_path.exists():
        return True
    async_client = download_dicom_data.async_client
    # 確保輸出目錄存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print('output_path',output_path)
    try:
        print('download_dicom_data.series_uid,',download_dicom_data.series_uid,)
        print('download_dicom_data.instances_uid,', download_dicom_data.instances_uid, )
        async with semaphore:
            instances_response = await async_client.get_instances_id_file(download_dicom_data.instances_uid,)
        await write_file(output_path, instances_response)
        return True
    except Exception as e:
        print(f"Error processing series {download_dicom_data.series_uid}: {str(e)}")
        traceback.print_exc()
        return False

async def main():
    download_tasks = []
    # UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    UPLOAD_DATA_DICOM_SEG_URL = "http://10.103.51.1:28042"
    async_client = AsyncOrthanc(UPLOAD_DATA_DICOM_SEG_URL,
                                username="radaxaiAdmin",
                                password="radaxaiAdmin666",
                                timeout=300.0)
    study_uid_list = [
        "58e16e38-2e7f9a4a-9c26a415-f55b1725-ffa1288d",
        # "dc7c15d1-881e24ab-7fbee4e7-f2eea96b-cd6fb500"
    ]
    output_directory = pathlib.Path("/home/david/ai-inference-dicom-file-testing")
    for study_uid in study_uid_list:
        # studies = await async_client.get_studies_id(study_uid)
        # series_uid_list = studies['Series']
        series_uid_list = [
            # T1
            "2c4bc8a9-2cd5576d-776c485f-180b3212-5c708eb1",
            # T2 FLAIR
            "a4e02ca3-f65e3730-4666e4da-992694b1-5a349090",
            # SWAN
            "fb9ec858-3790e92e-ddf5d73b-8cfcea20-63d5499f",
            # DWI
            "063e719-26189a06-6e7e1453-6f5fe3ef-972620f6",
            # ADC
            "38acec51-987b6c78-9dce8ee1-0f1379e3-57d61fe7"
        ]
        for series_uid in series_uid_list:
            series_response = await async_client.get_series_id(series_uid)
            series = await async_client.get_series_id(series_uid)
            instances_uid_list = series['Instances']
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
        #         break
        #     break
        # break
    results = await asyncio.gather(*download_tasks, return_exceptions=True)
    print('main results',results)

# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == "__main__":
    print("10000")
    asyncio.run(main())  # 使用asyncio.run來運行異步main函數
