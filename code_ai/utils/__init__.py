import pathlib
import re

# study_id_pattern = re.compile('^[0-9]{8}_[0-9]{8}_(MR|CT|CR|PR).*$', re.IGNORECASE)
study_id_pattern = re.compile('.*([0-9]{8,11}_[0-9]{8}_(MR|CT|PR|CR)_E?[0-9]{8,14})+.*', re.IGNORECASE)


def check_study_id(intput_path: pathlib.Path) -> bool:
    global study_id_pattern
    if intput_path.is_dir():
        result = study_id_pattern.match(intput_path.name)
        if result is not None:
            return True
    return False


def replace_suffix(filename: str, new_suffix: str, pattern=r'\.nii\.gz$|\.nii$'):
    return re.sub(pattern, new_suffix, filename)
