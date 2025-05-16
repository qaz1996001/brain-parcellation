#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import warnings
import aiohttp
import httpx
import orjson
import asyncio
import pydicom
warnings.filterwarnings("ignore")  # 忽略警告输出
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


async def upload_json_file(session, url, file_path):
    try:
        with open(file_path, 'rb') as f:
            data = orjson.loads(f.read())

        async with session.post(url, json=data) as response:
            response_text = await response.text()
            status = response.status
            print(f"Uploaded {file_path}: Status {status}")
            return {"file": file_path, "status": status, "response": response_text}
    except Exception as e:
        print(f"Error uploading {file_path}: {str(e)}")
        return {"file": file_path, "status": "error", "response": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="處理 AI predict 檔案至 web server")
    parser.add_argument('--Input', type=str, nargs='+',
                        default=' ',
                        help='json檔案')
    args = parser.parse_args()

    UPLOAD_DATA_JSON_URL = os.getenv("UPLOAD_DATA_JSON_URL")

    if isinstance(args.Input, str):
        with open(args.Input, 'rb') as f:
            data = orjson.loads(f.read())
        with httpx.Client() as client:
            response = client.post(UPLOAD_DATA_JSON_URL, json=data)
            print(f"Uploaded single file: Status {response.status_code}")

    if isinstance(args.Input, list):
        input_list = args.Input

        # Create a connection pool with limits
        conn = aiohttp.TCPConnector(limit=10)  # Limit concurrent connections

        # Create a client session
        async with aiohttp.ClientSession(connector=conn) as session:
            # Create tasks for all uploads
            tasks = [upload_json_file(session, UPLOAD_DATA_JSON_URL, file_path) for file_path in input_list]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Print summary
            success_count = sum(
                1 for result in results if isinstance(result.get("status"), int) and 200 <= result.get("status") < 300)
            print(f"Upload complete: {success_count}/{len(input_list)} files successfully uploaded")


if __name__ == '__main__':
    # asyncio.run(main())
    parser = argparse.ArgumentParser(description="處理 AI predict 檔案至 web server")
    parser.add_argument('--Input', type=str, nargs='+',
                        default=' ',
                        help='json檔案')
    args = parser.parse_args()

