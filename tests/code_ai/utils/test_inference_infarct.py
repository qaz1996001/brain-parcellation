from code_ai.utils.inference import build_inference_cmd
from code_ai.utils.inference.schema import InferenceEnum


def _setup_paths(tmp_path):
    path_root = tmp_path / "sean_root" / "sean"
    path_root.mkdir(parents=True)
    chuan_code = path_root.parent / "chuan" / "code"
    chuan_code.mkdir(parents=True)
    return path_root, chuan_code


def _create_study(tmp_path):
    study_name = "17900041_20251211_MR_21412040106"
    nifti_study_path = tmp_path / "nifti" / study_name
    dicom_study_path = tmp_path / "dicom" / study_name
    nifti_study_path.mkdir(parents=True)
    dicom_study_path.mkdir(parents=True)

    for filename in (
        "ADC.nii.gz",
        "DWI0.nii.gz",
        "DWI1000.nii.gz",
        "T2FLAIR_AXI.nii.gz",
    ):
        (nifti_study_path / filename).write_text("test")

    for folder in ("ADC", "DWI0", "DWI1000", "T2FLAIR_AXI"):
        (dicom_study_path / folder).mkdir(parents=True)

    return study_name, nifti_study_path, dicom_study_path


def _pick_item(cmd_items, inference_name: InferenceEnum):
    for item in cmd_items:
        name_value = str(item.name)
        if name_value == inference_name.value or name_value.endswith(
            f".{inference_name.value}"
        ):
            return item
    raise AssertionError(f"No cmd item found for {inference_name}")


def test_infarct_command_includes_multiple_dicom_dirs(tmp_path, monkeypatch):
    path_root, chuan_code = _setup_paths(tmp_path)
    monkeypatch.setenv("PATH_ROOT", str(path_root))
    study_name, nifti_study_path, dicom_study_path = _create_study(tmp_path)

    inference_cmd = build_inference_cmd(nifti_study_path, dicom_study_path)
    infarct_item = _pick_item(inference_cmd.cmd_items, InferenceEnum.Infarct)

    tokens = infarct_item.cmd_str.split()
    script_index = tokens.index(f"{chuan_code}/pipeline_infarct.sh")
    payload = tokens[script_index + 1 :]
    assert payload[0] == study_name
    input_count = len(infarct_item.input_list)
    assert payload[1 : 1 + input_count] == infarct_item.input_list
    dicom_segment = payload[1 + input_count : 1 + input_count + 3]
    assert dicom_segment == [
        str(dicom_study_path / "ADC"),
        str(dicom_study_path / "DWI0"),
        str(dicom_study_path / "DWI1000"),
    ]
    assert payload[1 + input_count + 3] == str(nifti_study_path)


def test_wmh_command_keeps_single_dicom_dir(tmp_path, monkeypatch):
    path_root, chuan_code = _setup_paths(tmp_path)
    monkeypatch.setenv("PATH_ROOT", str(path_root))
    study_name, nifti_study_path, dicom_study_path = _create_study(tmp_path)

    inference_cmd = build_inference_cmd(nifti_study_path, dicom_study_path)
    wmh_item = _pick_item(inference_cmd.cmd_items, InferenceEnum.WMH)

    tokens = wmh_item.cmd_str.split()
    script_index = tokens.index(f"{chuan_code}/pipeline_wmh.sh")
    payload = tokens[script_index + 1 :]
    assert payload[0] == study_name
    input_count = len(wmh_item.input_list)
    assert payload[1 : 1 + input_count] == wmh_item.input_list
    dicom_segment = payload[1 + input_count : -1]
    assert dicom_segment == [str(dicom_study_path / "T2FLAIR_AXI")]
    assert payload[-1] == str(nifti_study_path)
