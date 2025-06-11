"""
funboost现在 新增 命令行启动消费 发布  和清空消息


"""
import os
import sys
from pathlib import Path
from code_ai import load_dotenv
load_dotenv()
path_process = os.getenv("PATH_PROCESS")
path_json = os.getenv("PATH_JSON")
path_log = os.getenv("PATH_LOG")
# 建置資料夾
os.makedirs(path_process, exist_ok=True)  # 如果資料夾不存在就建立，
os.makedirs(path_json, exist_ok=True)  # 如果資料夾不存在就建立，
os.makedirs(path_log, exist_ok=True)  # 如果資料夾不存在就建立，

from code_ai.scheduler.scheduler_check_add_task import add_raw_dicom_to_nii_inference


project_root_path = Path(__file__).absolute().parent
print(f'project_root_path is : {project_root_path}  ,请确认是否正确')
sys.path.insert(1, str(project_root_path))  # 这个是为了方便命令行不用用户手动先 export PYTHONPATTH=项目根目录

# $$$$$$$$$$$$
# 以上的sys.path代码需要放在最上面,先设置好pythonpath再导入funboost相关的模块
# $$$$$$$$$$$$
import fire
from funboost.timing_job import ApsJobAdder
from funboost.core.cli.funboost_fire import BoosterFire, env_dict
from funboost import BoostersManager
from funboost.core.cli.discovery_boosters import BoosterDiscovery

# 需要启动的函数,那么该模块或函数建议建议要被import到这来, 否则需要要在 --import_modules_str 或 booster_dirs 中指定用户项目中有哪些模块包括了booster
'''
有4种方式,自动找到有@boost装饰器,注册booster

1. 用户亲自把要启动的消费函数所在模块或函数 手动 import 一下到此模块来
2. 用户在使用命令行时候 --import_modules_str 指定导入哪些模块路径,就能启动那些队列名来消费和发布了.
3. 用户使用BoosterDiscovery.auto_discovery_boosters  自动 import 指定文件夹下的 .py 文件来实现.
4  用户在使用命令行时候传参 project_root_path booster_dirs ,自动扫描模块,自动import
'''
env_dict['project_root_path'] = project_root_path


if __name__ == '__main__':

    # booster_dirs 用户可以自己增加扫描的文件夹,这样可以命令行少传了 --booster_dirs_str
    # BoosterDiscovery 可以多次调用
    BoosterDiscovery(project_root_path,
                     booster_dirs=['code_ai/task'], max_depth=1, py_file_re_str=None).auto_discovery()
    # 这个最好放到main里面,如果要扫描自身文件夹,没写正则排除文件本身,会无限懵逼死循环导入
    fire.Fire(BoosterFire, )
    aps_job_adder = ApsJobAdder(add_raw_dicom_to_nii_inference)

    # 先立即执行一次
    aps_job_adder.add_push_job(trigger='date')

    # 然后设置 cron 任务每 30 分钟执行一次
    aps_job_adder.add_push_job(
        trigger='cron',
        minute='*/30'  # 每 30 分钟执行一次
    )
    BoostersManager.multi_process_consume_all_queues(1)


'''
python /codes/funboost/funboost_cli_user.py   --booster_dirs_str=test_frame/test_funboost_cli/test_find_boosters --max_depth=2  push test_find_queue1 --x=1 --y=2
python /codes/funboost/funboost_cli_user.py   --booster_dirs_str=test_frame/test_funboost_cli/test_find_boosters --max_depth=2  consume test_find_queue1 
'''
