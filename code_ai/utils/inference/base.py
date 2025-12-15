import argparse
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from code_ai.utils import replace_suffix

from .schema import (
    Analysis,
    InferenceCmd,
    InferenceCmdItem,
    InferenceEnum,
    MRSeriesRenameEnum,
    T1SeriesRenameEnum,
    T2SeriesRenameEnum,
    Task,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
INFARCT_TARGET_SERIES = ("ADC", "DWI0", "DWI1000")

ENUM_CLASS_MAP = {
    "T1SeriesRenameEnum": T1SeriesRenameEnum,
    "T2SeriesRenameEnum": T2SeriesRenameEnum,
    "MRSeriesRenameEnum": MRSeriesRenameEnum,
    "InferenceEnum": InferenceEnum,
}


def _task_key(task_name: Union[InferenceEnum, str]) -> str:
    return task_name.value if isinstance(task_name, InferenceEnum) else str(task_name)


def _ensure_enum(task_name: Union[InferenceEnum, str]) -> InferenceEnum:
    return task_name if isinstance(task_name, InferenceEnum) else InferenceEnum(task_name)


def load_config(config_path: Optional[Union[str, os.PathLike[str]]] = None) -> Dict[str, Any]:
    path = pathlib.Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def get_enum_by_name(enum_name: str) -> Optional[Any]:
    if "." not in enum_name:
        return None
    enum_class_name, enum_value = enum_name.split(".", 1)
    enum_class = ENUM_CLASS_MAP.get(enum_class_name)
    if not enum_class:
        return None
    return getattr(enum_class, enum_value, None)


def resolve_enum_mapping_series(
    mapping_config: Dict[str, Sequence[Sequence[str]]],
) -> Dict[InferenceEnum, List[List[Any]]]:
    resolved_mapping: Dict[InferenceEnum, List[List[Any]]] = {}
    for model_name, series_list in mapping_config.items():
        try:
            model_enum = InferenceEnum(model_name)
        except ValueError:
            logger.warning("未知的推論模型：%s", model_name)
            continue
        resolved_series_list: List[List[Any]] = []
        for series in series_list or []:
            resolved_series: List[Any] = []
            for enum_name in series:
                enum_instance = get_enum_by_name(enum_name)
                if not enum_instance:
                    break
                resolved_series.append(enum_instance)
            if len(resolved_series) == len(series):
                resolved_series_list.append(resolved_series)
        if resolved_series_list:
            resolved_mapping[model_enum] = resolved_series_list
    return resolved_mapping


def get_file_list(
    input_path: pathlib.Path,
    suffixes: str,
    filter_name: Optional[str] = None,
) -> List[pathlib.Path]:
    if any(suffix in input_path.suffixes for suffix in suffixes):
        file_list = [input_path]
    else:
        file_list = sorted(input_path.rglob("*.nii*"))
    if filter_name:
        file_list = [path for path in file_list if filter_name in path.name]
    return file_list


def prepare_output_file_list(
    file_list: List[pathlib.Path],
    suffix: str,
    output_dir: Optional[pathlib.Path] = None,
) -> List[pathlib.Path]:
    outputs: List[pathlib.Path] = []
    for path in file_list:
        target_parent = output_dir if output_dir else path.parent
        outputs.append(target_parent.joinpath(replace_suffix(path.name, suffix)))
    return outputs


def _extract_basename_from_path(path: str) -> str:
    base_name = os.path.basename(path)
    if base_name.endswith(".nii.gz"):
        return base_name[:-7]
    if base_name.endswith(".nii"):
        return base_name[:-4]
    return base_name


def _build_dicom_series_path(
    dicom_study_path: pathlib.Path,
    nifti_study_path: pathlib.Path,
    series_basename: Optional[str],
) -> Optional[pathlib.Path]:
    if not series_basename:
        return None
    if dicom_study_path.name == nifti_study_path.name:
        return dicom_study_path.joinpath(series_basename)
    return dicom_study_path.joinpath(nifti_study_path.name, series_basename)


def _choose_default_infarct_basename(task: Task) -> Optional[str]:
    if not task.input_path_list:
        return None
    default_index = 1 if len(task.input_path_list) > 1 else 0
    return _extract_basename_from_path(task.input_path_list[default_index])


def _match_series_basename(candidate_names: List[str], target: str) -> Optional[str]:
    for name in candidate_names:
        if name == target:
            return name
    target_lower = target.lower()
    for name in candidate_names:
        if target_lower in name.lower():
            return name
    return None


def _resolve_infarct_dicom_inputs(
    task: Task,
    nifti_study_path: pathlib.Path,
    dicom_study_path: pathlib.Path,
    study_id: str,
) -> Tuple[Optional[str], Optional[List[str]]]:
    default_basename = _choose_default_infarct_basename(task)
    default_path = _build_dicom_series_path(dicom_study_path, nifti_study_path, default_basename)
    default_dir = str(default_path) if default_path else None

    candidate_names = [_extract_basename_from_path(path) for path in task.input_path_list]
    dicom_dirs: List[str] = []
    missing_reason: Optional[str] = None

    for series in INFARCT_TARGET_SERIES:
        matched_name = _match_series_basename(candidate_names, series)
        if not matched_name:
            missing_reason = f"missing series {series}"
            break
        dicom_path = _build_dicom_series_path(dicom_study_path, nifti_study_path, matched_name)
        if not dicom_path or not dicom_path.exists():
            missing_reason = f"dicom path not found for {matched_name}"
            break
        dicom_dirs.append(str(dicom_path))

    if missing_reason:
        logger.warning(
            "Falling back to single DICOM directory for infarct study %s: %s",
            study_id,
            missing_reason,
        )
        return default_dir, None

    return default_dir, dicom_dirs


def _resolve_dicom_inputs(
    key: InferenceEnum,
    task: Task,
    nifti_study_path: pathlib.Path,
    dicom_study_path: pathlib.Path,
    study_id: str,
) -> Tuple[Optional[str], Optional[List[str]]]:
    if key == InferenceEnum.Infarct:
        return _resolve_infarct_dicom_inputs(task, nifti_study_path, dicom_study_path, study_id)
    if not task.input_path_list:
        return None, None
    basename = _extract_basename_from_path(task.input_path_list[0])
    dicom_path = _build_dicom_series_path(dicom_study_path, nifti_study_path, basename)
    return (str(dicom_path) if dicom_path else None), None


def _iter_nifti_files(study_path: pathlib.Path) -> List[pathlib.Path]:
    return sorted(
        path
        for path in study_path.iterdir()
        if path.name.endswith(".nii") or path.name.endswith(".nii.gz")
    )


def check_study_mapping_inference(
    study_path: pathlib.Path,
    config_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Optional[Dict[str, Dict[str, List[str]]]]:
    config = load_config(config_path)
    model_mapping_series_dict = resolve_enum_mapping_series(config.get("model_mapping_series", {}))
    nifti_files = _iter_nifti_files(study_path)
    if not nifti_files:
        return None

    df_file = pd.DataFrame(nifti_files, columns=["file_path"])
    df_file["file_name"] = df_file["file_path"].map(lambda path: replace_suffix(path.name, ""))

    model_mapping_dict: Dict[str, List[str]] = {}
    for model_enum, model_mapping_series_list in model_mapping_series_dict.items():
        for mapping_series in model_mapping_series_list:
            expected_names = [enum_member.value for enum_member in mapping_series]
            result = np.intersect1d(df_file["file_name"], expected_names, return_indices=True)
            if result[0].shape[0] >= len(expected_names):
                df_result = df_file.iloc[result[1]]
                model_mapping_dict[model_enum.value] = [
                    str(path) for path in df_result["file_path"].to_list()
                ]
                break

    if not model_mapping_dict:
        return None
    return {study_path.name: model_mapping_dict}


def _render_once(template: str, base_output_path: str, task_name: Union[InferenceEnum, str]) -> str:
    file_name = template.format(task_name=_task_key(task_name))
    return os.path.join(base_output_path, file_name)


def _render_each_input(
    template: str,
    input_paths: List[str],
    base_output_path: str,
    task_name: Union[InferenceEnum, str],
) -> List[str]:
    task_value = _task_key(task_name)
    files: List[str] = []
    for input_path in input_paths:
        base_name = os.path.basename(input_path).split(".")[0]
        files.append(os.path.join(base_output_path, template.format(base_name=base_name, task_name=task_value)))
    return files


def _render_special_cmb(
    template: str,
    input_paths: List[str],
    base_output_path: str,
    task_name: Union[InferenceEnum, str],
) -> Optional[str]:
    if len(input_paths) < 2:
        return None
    base_name1 = os.path.basename(input_paths[0]).split(".")[0]
    base_name2 = os.path.basename(input_paths[1]).split(".")[0]
    swan_base = base_name1 if base_name1.startswith("SWAN") else base_name2
    other_base = base_name2 if base_name1.startswith("SWAN") else base_name1
    file_name = template.format(
        swan_base_name=swan_base,
        other_base_name=other_base,
        task_name=_task_key(task_name),
    )
    return os.path.join(base_output_path, file_name)


def _legacy_generate_output_files(
    input_paths: List[str],
    task_name: Union[InferenceEnum, str],
    base_output_path: str,
) -> List[str]:
    task_enum = _ensure_enum(task_name)
    files: List[str] = []
    match task_enum:
        case InferenceEnum.Aneurysm:
            files.extend(
                [
                    os.path.join(base_output_path, "Pred_Aneurysm.nii.gz"),
                    os.path.join(base_output_path, "Prob_Aneurysm.nii.gz"),
                    os.path.join(base_output_path, "Pred_Aneurysm_Vessel.nii.gz"),
                    os.path.join(base_output_path, "Pred_Aneurysm.json"),
                    os.path.join(base_output_path, "Pred_Aneurysm_Vessel16.nii.gz"),
                ]
            )
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split(".")[0]
                files.append(
                    os.path.join(
                        base_output_path,
                        f"synthseg_{base_name}_original_synthseg33.nii.gz",
                    )
                )
        case InferenceEnum.WMH_PVS:
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split(".")[0]
                files.append(
                    os.path.join(
                        base_output_path,
                        f"synthseg_{base_name}_original_{task_enum.value}.nii.gz",
                    )
                )
        case InferenceEnum.DWI:
            files.append(
                os.path.join(base_output_path, f"synthseg_DWI0_original_{task_enum.value}.nii.gz")
            )
        case InferenceEnum.CMB:
            base_name1 = os.path.basename(input_paths[0]).split(".")[0]
            base_name2 = os.path.basename(input_paths[1]).split(".")[0]
            if base_name1.startswith("SWAN"):
                files.append(
                    os.path.join(
                        base_output_path,
                        f"synthseg_{base_name1}_original_{task_enum.value}_from_"
                        f"synthseg_{base_name2}_original_{task_enum.value}.nii.gz",
                    )
                )
            else:
                files.append(
                    os.path.join(
                        base_output_path,
                        f"synthseg_{base_name2}_original_{task_enum.value}_from_"
                        f"synthseg_{base_name1}_original_{task_enum.value}.nii.gz",
                    )
                )
            files.append(os.path.join(base_output_path, "Pred_CMB.nii.gz"))
            files.append(os.path.join(base_output_path, "Pred_CMB.json"))
        case InferenceEnum.Area:
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split(".")[0]
                files.append(
                    os.path.join(
                        base_output_path,
                        f"synthseg_{base_name}_original_synthseg33.nii.gz",
                    )
                )
                files.append(
                    os.path.join(
                        base_output_path,
                        f"synthseg_{base_name}_original_synthseg.nii.gz",
                    )
                )
        case InferenceEnum.Infarct:
            files.extend(
                [
                    os.path.join(base_output_path, "Pred_Infarct.nii.gz"),
                    os.path.join(base_output_path, "Pred_Infarct_ADCth.nii.gz"),
                    os.path.join(base_output_path, "Pred_Infarct_synthseg.nii.gz"),
                    os.path.join(base_output_path, "Pred_Infarct.json"),
                ]
            )
        case InferenceEnum.WMH:
            files.extend(
                [
                    os.path.join(base_output_path, "Pred_WMH.nii.gz"),
                    os.path.join(base_output_path, "Pred_WMH_synthseg.nii.gz"),
                    os.path.join(base_output_path, "Pred_WMH.json"),
                ]
            )
        case _:
            pass
    return files


def generate_output_files(
    input_paths: List[str],
    task_name: Union[InferenceEnum, str],
    base_output_path: str,
    config_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> List[str]:
    config = load_config(config_path)
    task_formats = config.get("output_formats", {}).get(_task_key(task_name), [])
    if not task_formats:
        return _legacy_generate_output_files(input_paths, task_name, base_output_path)

    output_files: List[str] = []
    for format_spec in task_formats:
        template = format_spec.get("template", "")
        apply_to = format_spec.get("apply_to", "once")
        special = format_spec.get("special")

        if special == "swan_detection":
            special_file = _render_special_cmb(template, input_paths, base_output_path, task_name)
            if special_file:
                output_files.append(special_file)
            continue

        if apply_to == "each_input":
            output_files.extend(_render_each_input(template, input_paths, base_output_path, task_name))
            continue

        output_files.append(_render_once(template, base_output_path, task_name))

    return output_files


def build_Area(mode, file_dict) -> Tuple[argparse.Namespace, List[pathlib.Path]]:
    parser = argparse.ArgumentParser(prog="build_Area")
    args = parser.parse_known_args()[0]
    args.cmb = False
    args.wmh = False
    args.dwi = False
    args.wm_file = False
    args.all = False
    args.depth_number = 5
    setattr(args, mode, True)

    output_path = pathlib.Path(file_dict["output_path"])
    args.intput_file_list = [pathlib.Path(path) for path in file_dict["input_path_list"]]
    args.resample_file_list = prepare_output_file_list(args.intput_file_list, "_resample.nii.gz", output_path)
    args.synthseg_file_list = prepare_output_file_list(args.resample_file_list, "_synthseg.nii.gz", output_path)
    args.synthseg33_file_list = prepare_output_file_list(args.resample_file_list, "_synthseg33.nii.gz", output_path)
    args.david_file_list = prepare_output_file_list(args.resample_file_list, "_david.nii.gz", output_path)
    args.wm_file_list = prepare_output_file_list(args.resample_file_list, "_wm.nii.gz", output_path)
    return args, args.intput_file_list


def get_synthseg_args_file(inference_name, file_dict) -> Tuple[Optional[argparse.Namespace], Optional[List[pathlib.Path]]]:
    output_path = pathlib.Path(file_dict["output_path"])
    match inference_name:
        case InferenceEnum.Area | InferenceEnum.Aneurysm:
            return build_Area("wm_file", file_dict)
        case InferenceEnum.WMH_PVS:
            args, file_list = build_Area("wmh", file_dict)
            args.wmh_file_list = prepare_output_file_list(args.resample_file_list, "_WMHPVS.nii.gz", output_path)
            return args, file_list
        case InferenceEnum.DWI:
            args, file_list = build_Area("dwi", file_dict)
            args.dwi_file_list = prepare_output_file_list(args.resample_file_list, "_DWI.nii.gz", output_path)
            return args, file_list
        case InferenceEnum.CMB:
            args, file_list = build_Area("cmb", file_dict)
            args.cmb_file_list = prepare_output_file_list(args.resample_file_list, "_CMB.nii.gz", output_path)
            return args, file_list
        case _:
            return None, None


def _apply_post_process_rules(
    processed_paths: List[str],
    post_process_config: Dict[str, Any],
) -> bool:
    if not post_process_config:
        return False

    condition = post_process_config.get("condition", {})
    expected_inputs = condition.get("input_match", [])
    if expected_inputs:
        normalized_expected = []
        for expected in expected_inputs:
            enum_instance = get_enum_by_name(expected)
            normalized_expected.append(enum_instance.value if enum_instance else expected)
        normalized_inputs = [replace_suffix(os.path.basename(path), "") for path in processed_paths]
        result = np.intersect1d(normalized_inputs, normalized_expected, return_indices=True)
        if result[0].shape[0] != len(normalized_expected):
            return False

    modified = False
    for file_spec in post_process_config.get("add_files", []):
        replace_spec = file_spec.get("replace", {})
        source = replace_spec.get("from")
        target = replace_spec.get("to")
        if not source or not target:
            continue
        for path in list(processed_paths):
            if source in path:
                processed_paths.append(path.replace(source, target))
                modified = True
                break
    return modified


def _legacy_input_post_process(
    input_paths: List[str],
    model_name: Union[InferenceEnum, str],
) -> List[str]:
    processed_paths = list(input_paths)
    inference_key = _ensure_enum(model_name)
    base_names = [replace_suffix(os.path.basename(path), "") for path in processed_paths]

    if inference_key == InferenceEnum.Infarct and set(INFARCT_TARGET_SERIES).issubset(set(base_names)):
        adc_path = next((path for path in processed_paths if path.endswith("ADC.nii.gz")), None)
        if adc_path:
            processed_paths.append(adc_path.replace("ADC.nii.gz", "synthseg_DWI0_original_DWI.nii.gz"))
        return processed_paths

    if inference_key == InferenceEnum.WMH and any(name.startswith("T2FLAIR_AXI") for name in base_names):
        flair_path = next((path for path in processed_paths if "T2FLAIR_AXI.nii.gz" in path), None)
        if flair_path:
            processed_paths.append(
                flair_path.replace("T2FLAIR_AXI.nii.gz", "synthseg_T2FLAIR_AXI_original_synthseg5.nii.gz")
            )
            processed_paths.append(
                flair_path.replace("T2FLAIR_AXI.nii.gz", "synthseg_T2FLAIR_AXI_original_WMH_PVS.nii.gz")
            )
        return processed_paths

    return processed_paths


def build_input_post_process(
    input_paths: List[str],
    model_name: Union[InferenceEnum, str],
    config_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> List[str]:
    processed_paths = list(input_paths)
    config = load_config(config_path)
    post_process_config = config.get("input_post_process", {}).get(_task_key(model_name))
    if post_process_config and _apply_post_process_rules(processed_paths, post_process_config):
        return processed_paths
    return _legacy_input_post_process(processed_paths, model_name)


def build_analysis(
    study_path: pathlib.Path,
    config_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Analysis:
    mapping_inference = check_study_mapping_inference(study_path, config_path)
    study_id = study_path.name
    if not mapping_inference:
        return Analysis(study_id=study_id)

    tasks: Dict[str, Task] = {}
    task_dict = next(iter(mapping_inference.values()), {})
    for model_name, input_paths in task_dict.items():
        inference_key = _ensure_enum(model_name)
        processed_inputs = build_input_post_process(list(input_paths), inference_key, config_path)
        task_output_files = generate_output_files(
            processed_inputs,
            inference_key,
            str(study_path),
            config_path,
        )
        task = Task(
            intput_path_list=processed_inputs,
            output_path=str(study_path),
            output_path_list=task_output_files,
        )
        tasks[inference_key.value] = task

    return Analysis(study_id=study_id, **tasks)


def build_inference_cmd(
    nifti_study_path: pathlib.Path,
    dicom_study_path: pathlib.Path,
    config_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> InferenceCmd:
    from code_ai.pipeline import pipelines

    analysis = build_analysis(nifti_study_path, config_path)
    inference_item_list: List[InferenceCmdItem] = []

    for key, value in analysis.model_dump().items():
        if key == "study_id" or value is None:
            continue
        try:
            enum_key = InferenceEnum(key)
        except ValueError:
            continue
        pipeline = pipelines.get(enum_key)
        if not pipeline:
            continue
        task = getattr(analysis, key)
        input_dicom_dir, input_dicom_dirs = _resolve_dicom_inputs(
            enum_key,
            task,
            nifti_study_path,
            dicom_study_path,
            analysis.study_id,
        )
        cmd_str = pipeline.generate_cmd(
            analysis.study_id,
            task,
            input_dicom_dir=input_dicom_dir,
            input_dicom_dirs=input_dicom_dirs,
        )
        inference_item_list.append(
            InferenceCmdItem(
                study_id=analysis.study_id,
                name=enum_key,
                cmd_str=cmd_str,
                input_list=task.input_path_list,
                output_list=task.output_path_list,
                input_dicom_dir=input_dicom_dir or "",
            )
        )

    return InferenceCmd(cmd_items=inference_item_list)

