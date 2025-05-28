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