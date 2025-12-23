#!/bin/bash

export PYTHONPATH=$(dirname $(dirname "$(cd "$(dirname "$0")" && pwd)"))
ENVENVENV_PATH="$PYTHONPATH/.env"

echo "PYTHONPATH:$PYTHONPATH"
if [ -f "$ENV_PATH" ]; then
    source "$ENVENVENV_PATH"
else
    echo "Warning: .env file not found at $ENVENVENV_PATH"
fi
echo "ENVENVENV_PATH:$ENVENVENV_PATH"
echo "PYTHON3:$PYTHON3"
echo "FSL_FLIRT:$FSL_FLIRT"
