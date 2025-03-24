import os
import sys
sys.path.append(os.path.abspath('.'))
print('path', sys.path)
import argparse
import pathlib
from celery import Celery


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dicom', dest='input_dicom', type=str,
    #                     help="input the rename dicom folder.\r\n")
    parser.add_argument('-i', '--input_nifti', dest='input_nifti', type=str,
                        help="input the rename nifti folder.\r\n",
                        default='/mnt/d/00_Chen/Task04_git/data_0106/')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help="result output folder.\r\n",
                        default='/mnt/d/00_Chen/Task04_git/data_0106')
    return parser.parse_args()


def main():
    args = parse_arguments()
    app = Celery('tasks',
                 broker='pyamqp://guest:guest@localhost:5672/celery',
                 backend='redis://localhost:10079/1'
                 )
    app.config_from_object('code_ai.celery_config')
    input_dir = pathlib.Path(args.input_nifti)
    output_dir = pathlib.Path(args.output)
    result = app.send_task('code_ai.task.workflow.task_inference_workflow',
                           args=(args.input_nifti,
                                 args.output),
                           queue='default',
                           routing_key='celery')





if __name__ == '__main__':
    main()