#!/bin/bash
#export PYTHONPATH=$(dirname $(dirname "$PWD"))
#echo PYTHONPATH
export PYTHONPATH=$(dirname $(dirname "$(cd "$(dirname "$0")" && pwd)"))
echo "PYTHONPATH:$PYTHONPATH"
python3 "$PYTHONPATH/code_ai/pipeline/pipeline_dicom_to_nii.py" "$@"

