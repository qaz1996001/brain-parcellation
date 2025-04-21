#!/bin/bash
#export PYTHONPATH=$(dirname $(dirname "$PWD"))
#echo PYTHONPATH
export PYTHONPATH=$(dirname $(dirname "$(cd "$(dirname "$0")" && pwd)"))
echo "PYTHONPATH:$PYTHONPATH"
python3 "$PYTHONPATH/code_ai/pipeline/pipeline_dicom_to_nii.py" "$@"

## bash 00_Chen/Task04_git/code_ai/pipeline/pipeline_dicom_to_nii.sh --input_dicom /mnt/e/raw_dicom/  --output_dicom /mnt/e/rename_dicom_20250421
## bash 00_Chen/Task04_git/code_ai/pipeline/pipeline_dicom_to_nii.sh --input_dicom /mnt/e/raw_dicom/02695350_21210300104  --output_dicom /mnt/e/rename_dicom_20250421
## bash 00_Chen/Task04_git/code_ai/pipeline/pipeline_dicom_to_nii.sh --input_dicom /mnt/e/raw_dicom/02695350_21210300104  --output_dicom /mnt/e/rename_dicom_20250421/02695350_21210300104 --output_nifti /mnt/e/rename_nifti_20250421/02695350_21210300104
#bash 00_Chen/Task04_git/code_ai/pipeline/pipeline_dicom_to_nii.sh --input_dicom /mnt/e/raw_dicom/02695350_21210300104  --output_dicom /mnt/e/rename_dicom_20250421/202504211716 --output_nifti /mnt/e/rename_nifti_20250421/202504211716
                                                                                                                                                                                           #

