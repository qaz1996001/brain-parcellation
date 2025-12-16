from code_ai.utils import replace_suffix

from .base import (
    build_Area,
    build_analysis,
    build_inference_cmd,
    build_input_post_process,
    check_study_mapping_inference,
    generate_output_files,
    get_enum_by_name,
    get_file_list,
    get_synthseg_args_file,
    load_config,
    prepare_output_file_list,
    resolve_enum_mapping_series,
)
from .schema import Analysis, InferenceCmd, InferenceCmdItem, InferenceEnum, Task

__all__ = [
    "Analysis",
    "InferenceCmd",
    "InferenceCmdItem",
    "InferenceEnum",
    "Task",
    "build_Area",
    "build_analysis",
    "build_inference_cmd",
    "build_input_post_process",
    "check_study_mapping_inference",
    "generate_output_files",
    "get_enum_by_name",
    "get_file_list",
    "get_synthseg_args_file",
    "load_config",
    "prepare_output_file_list",
    "replace_suffix",
    "resolve_enum_mapping_series",
]
