## new venv install

``` bash
pip install uv 
git clone ....
cd ./brain-parcellation
uv sync
source .venv/bin/activate
export PYTHONPATH=$(pwd) && python3 backend/app/main.py
```


## system python install

``` bash
pip install uv
git clone ....
cd ./brain-parcellation
uv pip install -r pyproject.toml --system
export PYTHONPATH=$(pwd) && python3 backend/app/main.py

conda activate shhai
conda deactivate
```

```bash 
conda activate tf_2_14

cd ./brain-parcellation && conda activate tf_2_14 && export PYTHONPATH=$(pwd) &&  python3 backend/app/main.py
cd ./brain-parcellation && conda activate tf_2_14 && export PYTHONPATH=$(pwd) &&  python3 funboost_cli_user.py


conda activate tf_2_14

```

#### MRA_BRAIN 

```bash 
cd ./brain-parcellation

```

```bash 
export PYTHONPATH=$(pwd) &&  python3 code_ai/pipeline/pipeline_aneurysm_tensorflow.py \
 --ID 14914694_20220905_MR_21109050071 \
  --Inputs /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/MRA_BRAIN.nii.gz \
   --Output_folder /mnt/e/pipeline/sean/rename_nifti \
   --InputsDicomDir /mnt/e/pipeline/sean/rename_dicom/14914694_20220905_MR_21109050071/MRA_BRAIN
```
#### WMH

```bash 
export PYTHONPATH=$(pwd) &&  python3 code_ai/pipeline/pipeline_synthseg_wmh_tensorflow.py \
 --ID 14914694_20220905_MR_21109050071 \
  --Inputs /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/T2FLAIR_AXI.nii.gz \
   --Output_folder /mnt/e/pipeline/sean/rename_nifti \
   --InputsDicomDir /mnt/e/pipeline/sean/rename_dicom/14914694_20220905_MR_21109050071/T2FLAIR_AXI
```
```bash 
export PYTHONPATH=$(pwd) &&  python3 code_ai/pipeline/pipeline_wmh_tensorflow.py \
 --ID 14914694_20220905_MR_21109050071 \
  --Inputs /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/T2FLAIR_AXI.nii.gz \
  /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/synthseg_T2FLAIR_AXI_original_WMH_PVS.nii.gz \
   /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/synthseg_T2FLAIR_AXI_original_synthseg5.nii.gz \
   --Output_folder /mnt/e/pipeline/sean/rename_nifti \
   --InputsDicomDir /mnt/e/pipeline/sean/rename_dicom/14914694_20220905_MR_21109050071/T2FLAIR_AXI
```
```bash 
sudo apt install unzip
conda activate tf_2_14
uv pip install -r pyproject.toml --system

cd /mnt/d/00_Chen/Task04_git && conda activate tf_2_14 && export PYTHONPATH=$(pwd) &&  python3 backend/app/main.py
cd /mnt/d/00_Chen/Task04_git && conda activate tf_2_14 && export PYTHONPATH=$(pwd) &&  python3 funboost_cli_user.py


conda activate tf_2_14

cat <<EOF | xargs -I{} bash -c 'conda run -n tf_2_14 bash -c "
cd /var/www/brain-parcellation && \
&export PYTHONPATH=\$(pwd) & \
python3 code_ai/pipeline/raw_diom_to_nii_inference.py \
  --input_dicom \"{}\" \
  --output_dicom /data/4TB1/pipeline/sean/rename_dicom \
  --output_nifti /data/4TB1/pipeline/sean/rename_nifti" &' 
/data/4TB1/raw_dicom/ai_team/sync_orthanc/5cbf750a-90427da9-fca347a6-45ca1732-6893b6c6
/data/4TB1/raw_dicom/ai_team/sync_orthanc/8cfadfd6-fe04d6c7-2d1f2c12-9e852e40-ef6777c7
/data/4TB1/raw_dicom/ai_team/sync_orthanc/9c4a93b7-b06654b8-dfe4c530-607a39d1-b75f0672
/data/4TB1/raw_dicom/ai_team/sync_orthanc/45fd38e7-8822c36d-43d1302c-77d831de-d9dbc470
/data/4TB1/raw_dicom/ai_team/sync_orthanc/1823ce5c-7fc2efe7-fb7c042a-9fe3edea-0b1660d9
/data/4TB1/raw_dicom/ai_team/sync_orthanc/5069a078-75659a75-78bbdf0d-4adf4b42-d3981727
/data/4TB1/raw_dicom/ai_team/sync_orthanc/bb7d7abf-77552121-92b730aa-d874dfcb-1b026701
/data/4TB1/raw_dicom/ai_team/sync_orthanc/cb71cc54-a9d2d7c0-8e7b0b76-e222a8f5-50440dd4
/data/4TB1/raw_dicom/ai_team/sync_orthanc/d6c3de1b-4356ed70-8cd30c60-5751ea45-f52934c0

EOF

```
```
cat <<EOF | xargs -I{} bash -c 'conda run -n tf_2_14 bash -c "
cd /home/david/brain-parcellation && \
export PYTHONPATH=\$(pwd) && \
python3 code_ai/pipeline/raw_diom_to_nii_inference.py \
  --input_dicom \"{}\" \
  --output_dicom /home/david/pipeline/sean/rename_dicom \
  --output_nifti /home/david/pipeline/sean/rename_nifti" &' 
/home/david/pipeline/tmp/021bed3e-f9be3f49-e8159e42-f9c4f2c8-737fb79c

EOF

```


### 

#### 第一步：建立
```
mkdir test
cd test
cp -r ../brain-parcellation ./
git reset --hard
git fetch
git checkout rad_ai_infer_test
git pull
```
##### 1.2 變更  .env
``` 

PATH_ROOT=/mnt/e/pipeline/test/sean

PATH_RAW_DICOM=/mnt/e/pipeline/test/raw_dicom


REDIS_HOST=127.0.0.1
REDIS_USERNAME=
REDIS_PASSWORD=
REDIS_PORT=10079
REDIS_DB=2
REDIS_DB_FILTER_AND_RPC_RESULT=3
REDIS_DB_FASTAPI_CACHE=7


UPLOAD_DATA_HOST=127.0.0.1
UPLOAD_DATA_DICOM_SEG_PORT=8042
UPLOAD_DATA_DICOM_SEG_URL="http://${UPLOAD_DATA_HOST}:${UPLOAD_DATA_DICOM_SEG_PORT}"

UPLOAD_DATA_JSON_PORT=7999
UPLOAD_DATA_API_URL="http://${UPLOAD_DATA_HOST}:${UPLOAD_DATA_JSON_PORT}/api/v1"

# AI_APP  #################################
AI_APP_PORT=7999
AI_APP_TITLE="SHH AI API TEST environments"
AI_APP_DESCRIPTION="API FOR SHH AI TEST environments"
AI_APP_VERSION="1.1.0"
AI_APP_CONNECTION_STRING="postgresql+asyncpg://postgres_n:postgres_p@127.0.0.1:15433/dicom_testing"
```

#### 第二步：關閉連接
```bash
docker exec -it db_server psql -U postgres_n -d postgres -c "
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'dicom' AND pid <> pg_backend_pid();"
```
#### 第三步：建立新資料庫（分開執行）
```bash
docker exec -it db_server psql -U postgres_n -d postgres -c "CREATE DATABASE dicom_testing WITH TEMPLATE dicom OWNER postgres_n;"

```
#### 第四步：驗證DB內容

#### 第五步：清空測試DB

```bash
docker exec -it db_server psql -U postgres_n -d postgres -c "truncate table dcop_event_bt;"
docker exec -it db_server psql -U postgres_n -d postgres -c "truncate table dcop_event_bth;"
```

/openspec:apply 繼續  add-environment-support  請分析 backend、code_ai有用到環境變數的程式碼，與對應的程式調用流程。我需要將程式碼都改成pure function，減少side effect，非必要不要有side effect。
我想要讓每一個調用鍊都有以下的 log 方式，log 設定需要有通用含數，每一個調用鍊有自己的log檔案