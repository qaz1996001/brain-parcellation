#!/bin/bash
#export PYTHONPATH=$(dirname $(dirname "$PWD"))
#echo PYTHONPATH
export PYTHONPATH=$(dirname $(dirname "$(cd "$(dirname "$0")" && pwd)"))
echo "PYTHONPATH:$PYTHONPATH"
python3 "$PYTHONPATH/code_ai/pipeline/pipeline_raw_diom_to_nii_inference.py" "$@"