import json
import logging
import re
import shutil
import os
import pathlib
from typing import List

import httpx
import requests

from code_ai import load_dotenv
from backend.app.sync import urls as sync_urls


if __name__ == '__main__':
    result_parent_set = {('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/7343d7a3-dbd8985b-83cb9baf-a6a82f09-b81c0a0f',
                          '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/ADC'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/892668ed-e963f615-f0195a24-3d94eb0c-716a0d68',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBECE_SAGr'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/308454c5-d2ff7ec1-74ce99a4-4281ba1b-a3a1d20b',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/DWI0'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/767b7a86-3609ad08-41f1e8bd-c2ffcd1e-d52c404f',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/MRA_BRAIN'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/c411a7cd-00b1a6c9-9f294a8d-55e21f32-ed24927f',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/MRAVR_NECK'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/892668ed-e963f615-f0195a24-3d94eb0c-716a0d68',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBECE_CORr'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/8cfeeef5-d528b61a-7163c714-5f8dc536-84b3378f',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBE_SAG'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/d66c9459-b7096a5c-ebe75fff-aee4b22f-ce9e8ea3',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBE_SAGr'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/d3d8366b-9c601d4c-277f4ec6-ff7d14f4-14a0c7d0',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBECE_SAG'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/480d0af9-1b354283-1275c5a9-779d1d19-f5badec3',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/SWAN'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/4a128011-eb094052-044f0a9b-dad8e303-d1bd2514',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T2_COR'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/88fc5a1e-70f3a268-83359e24-b7d02988-7597e83e',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBE_AXIr'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/580ada15-b16f5062-e11479b7-3cc015a5-ca9d5b71',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBECE_AXIr'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/acbbeffd-0e3771a9-6df1df86-fe5a1c68-445cdc30',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/DSCMTT_COLOR'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/0c0a1444-9238e5ad-fbcd0251-335322e7-9af7b058',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1FLAIR_AXI'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/f65fe67c-f5f69656-73afb17b-b0c465a1-6bc7f053',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/DSCCBF_COLOR'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/5828992d-127486bb-31d8ffb4-b6303e9e-bcc8127d',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/MRA_NECK'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/ec07e170-f167ee5d-503ee7cd-becd8038-c48dedab',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/SWANPHASE'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/30b67051-ea598840-3d7e18d5-7ca18e99-b5812b64',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/MRAVR_BRAIN'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/fc340cd5-37fe175d-ed6feb62-bab88fdf-9850e140',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T2FLAIR_AXI'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/5a01d8bd-a9b020b4-508e6b1f-55c112ef-cdb28595',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/SWAN'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/864b5235-e269bd92-be2fb3b3-43b18f0f-bf580226',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/DSCCBV_COLOR'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/580ada15-b16f5062-e11479b7-3cc015a5-ca9d5b71',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBECE_SAGr'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/d66c9459-b7096a5c-ebe75fff-aee4b22f-ce9e8ea3',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBE_CORr'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/31fb1be1-71d25700-b131126f-c73708af-42d28093',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/MRAVR_BRAIN'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/86364c14-51867d5e-4ecffab6-36054e99-ad1ff077',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/ADC'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/88fc5a1e-70f3a268-83359e24-b7d02988-7597e83e',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/T1CUBE_SAGr'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/4fd61075-0d21921e-6f493a03-06f34542-791c1ef8',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/DSC'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/308454c5-d2ff7ec1-74ce99a4-4281ba1b-a3a1d20b',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/DWI1000'),
     ('/mnt/e/raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/5a01d8bd-a9b020b4-508e6b1f-55c112ef-cdb28595',
      '/mnt/e/pipeline/sean/rename_dicom/10089413_20210201_MR_21002010079/SWANmIP')}

    load_dotenv()
    UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
    json_data = json.dumps(sorted(result_parent_set))

    url = "{}/sync{}".format(UPLOAD_DATA_API_URL,
                       sync_urls.SERIES_PROT_STUDY)
    print(url)

    client = httpx.Client()
    with httpx.Client() as client:
        response = client.post(url,
                               data=json_data)
    print(response)
    print(response.text)
