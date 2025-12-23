import argparse
import pathlib
import subprocess
from typing import List

if __name__ == '__main__':
    from code_ai.utils_inference import InferenceEnum
    from code_ai.utils_synthseg import TemplateProcessor
    from code_ai.utils_inference import replace_suffix

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_mode', type=str, required=True,
                        help='save_mode')
    parser.add_argument('--cmb_file_list', type=str,nargs='+',
                        help='cmb_file_list')

    args = parser.parse_args()
    save_mode = args.save_mode
    if save_mode == InferenceEnum.CMB:
        cmb_file_list = list(map(lambda cmd_file :pathlib.Path(cmd_file),args.cmb_file_list))
        original_cmb_file_list: List[pathlib.Path] = list(map(lambda x:x.parent.joinpath(f"synthseg_{x.name.replace('resample', 'original')}"),cmb_file_list))
        if original_cmb_file_list[0].name.startswith('synthseg_SWAN'):
            swan_file = original_cmb_file_list[0]
            t1_file = original_cmb_file_list[1]
        else:
            swan_file = original_cmb_file_list[1]
            t1_file = original_cmb_file_list[0]

        template_basename: str = replace_suffix(t1_file.name, '')
        synthseg_basename: str = replace_suffix(swan_file.name, '')
        print('template_basename',template_basename)
        print('synthseg_basename',synthseg_basename)

        template_coregistration_file_name = swan_file.parent.joinpath(f'{synthseg_basename}_from_{template_basename}')
        cmd_str = TemplateProcessor.flirt_cmd_base.format(t1_file,swan_file,template_coregistration_file_name)
        # apply_cmd_str = TemplateProcessor.flirt_cmd_apply.format(t1_file, swan_file, template_coregistration_file_name)
        print('cmd_str',cmd_str)
        process = subprocess.Popen(args=cmd_str, cwd='/', shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print('stdout',stdout)
        print('stderr',stderr)