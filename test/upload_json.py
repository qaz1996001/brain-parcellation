import argparse

import httpx
import orjson

def upload_json_aiteam(json_file):
    UPLOAD_DATA_JSON_URL = 'http://localhost:84/api/ai_team/create_or_update'
    client = httpx.Client()
    if isinstance(json_file, str):
        with open(json_file, 'rb') as f:
            data = orjson.loads(f.read())
        with client:
            response = client.post(UPLOAD_DATA_JSON_URL, json=data)
            print(f"Uploaded single file: Status {response.status_code}")

    if isinstance(json_file, list):
        input_list = json_file
        with client:
            for inputs in input_list:
                with open(inputs, 'rb') as f:
                    data = orjson.loads(f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--Inputs', type=str, nargs='+',
                        help='用於輸入平台的JOSN檔案')
    args = parser.parse_args()
    for json_file in args.Inputs:
        upload_json_aiteam(json_file)
