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



``` bash
cd /var/www/brain-parcellation && conda activate tf_2_14 && export PYTHONPATH=$(pwd) && python3 backend/app/main.py
cd /var/www/brain-parcellation && conda activate tf_2_14 && export PYTHONPATH=$(pwd) && python3 funboost_cli_user.py

cd /var/www/brain-parcellation
conda activate tf_2_14


conda deactivate
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

conda activate tf_2_14

cat <<EOF | xargs -I{} bash -c 'conda run -n tf_2_14 bash -c "
cd /var/www/brain-parcellation && \
export PYTHONPATH=\$(pwd) && \
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
```bash 
export PYTHONPATH=$(pwd) &&  python3 funboost_cli_user.py
```
### 

1. raw dicom -> rename dicom 
   1. DCOPEventDicomService.get_series_info
   2. dicom_to_nii - > process_dir
   3. DCOPEventDicomService.post_ope_no_task 
   4. DCOPEventDicomService.check_study_series_transfer_complete

2. rename dicom -> rename nifti 
   1. DCOPEventDicomService.check_study_series_transfer_complete
   2. dicom_to_nii - > dicom_2_nii_file
   3. DCOPEventDicomService.study_series_nifti_tool 
   4. DCOPEventDicomService.check_study_series_conversion_complete
   
3. rename nifti -> pipeline_inference 
   1. DCOPEventDicomService.check_study_series_conversion_complete 
   2. 
   3. DCOPEventDicomService.study_series_inference_nifti_tool
   4. DCOPEventDicomService.check_study_series_inference_complete
4. pipeline_inference -> upload 
   1. DCOPEventDicomService.check_study_upload_complete 

## Suntory
1. raw dicom -> rename dicom  > rename nifti \
   [raw_diom_to_nii_inference.py](code_ai/pipeline/raw_diom_to_nii_inference.py)

```bash
export PYTHONPATH=$(pwd) && python3 code_ai/pipeline/raw_diom_to_nii_inference.py \
  --input_dicom /data/10TB/sean/Suntory/raw_data \
  --output_dicom /data/10TB/sean/Suntory/rename_dicom \
  --output_nifti /data/10TB/sean/Suntory/rename_nifti

```
2. nifti -> synthseg_wmh_tensorflow