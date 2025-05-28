import os
import pathlib


def setup_pip_tf_cuda_env() -> bool:
    try:
        import nvidia
        nvidia_path = pathlib.Path(nvidia.__file__).parent
        so_file = sorted(nvidia_path.rglob('*.so.*'))
        bc_file = sorted(nvidia_path.rglob('*.bc'))
        so_file.extend(bc_file)
        bin_file = sorted(nvidia_path.rglob('bin'))
        so_dir = list(set(map(lambda x: x.parent, so_file)))
        so_dir.extend(bin_file)
        so_dir.extend(bc_file)
        # PATH_STR = os.environ['PATH']
        # PATH_STR = "{}:{}".format(PATH_STR,':'.join(map(lambda x:str(x),so_dir)))
        # print('PATH_STR',PATH_STR)
        # os.environ['PATH'] = PATH_STR
        # PATH_STR = os.environ['PATH']

        PATH_STR = os.environ['PATH']
        PATH_STR = "{}:{}".format(PATH_STR, ':'.join(map(lambda x: str(x), bc_file)))
        os.environ['PATH'] = PATH_STR
        LD_LIBRARY_PATH = "{}".format(':'.join(map(lambda x: str(x), so_dir)))
        # print('LD_LIBRARY_PATH',LD_LIBRARY_PATH)
        os.environ['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH
        # XLA_FLAGS = "{}:{}".format(str(bc_file[0].parent.parent.parent), ':'.join(map(lambda x: str(x), so_dir)))
        XLA_FLAGS = "{}".format(str(bc_file[0].parent.parent.parent),)
        os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir={}".format(XLA_FLAGS)
        return True
    except ImportError:
        pass
    return False
