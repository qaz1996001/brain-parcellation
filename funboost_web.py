"""
funboost现在 新增 命令行启动消费 发布  和清空消息


"""
import os
import sys
from pathlib import Path
from code_ai import load_dotenv
from funboost.core.cli.funboost_fire import  env_dict
load_dotenv()

project_root_path = Path(__file__).absolute().parent
print(f'project_root_path is : {project_root_path}  ,请确认是否正确')
sys.path.insert(1, str(project_root_path))  # 这个是为了方便命令行不用用户手动先 export PYTHONPATTH=项目根目录

# $$$$$$$$$$$$
# 以上的sys.path代码需要放在最上面,先设置好pythonpath再导入funboost相关的模块
# $$$$$$$$$$$$

env_dict['project_root_path'] = project_root_path


if __name__ == '__main__':

    # booster_dirs 用户可以自己增加扫描的文件夹,这样可以命令行少传了 --booster_dirs_str
    # BoosterDiscovery 可以多次调用
    from funboost.funboost_web_manager.app import start_funboost_web_manager
    start_funboost_web_manager()

