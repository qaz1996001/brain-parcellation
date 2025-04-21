#!/bin/bash
#export PYTHONPATH=$(dirname $(dirname "$PWD"))
#echo PYTHONPATH
export PYTHONPATH=$(dirname $(dirname "$(cd "$(dirname "$0")" && pwd)"))
echo "PYTHONPATH:$PYTHONPATH"
python3 "$PYTHONPATH/code_ai/pipeline/pipeline_raw_diom_to_nii_inference.py" "$@"

# /mnt/d/00_Chen/Task04_git/code_ai/pipeline/pipeline_cmb_tensorflow.sh --ID 02695350_20240109_MR_21210300104 --Inputs /mnt/e/rename_nifti_20250421/202504211716/02695350_20240109_MR_21210300104/T1BRAVO_AXI.nii.gz /mnt/e/rename_nifti_20250421/202504211716/02695350_20240109_MR_21210300104/SWAN.nii.gz --Output_folder  /mnt/e/rename_nifti_20250421/202504211716/