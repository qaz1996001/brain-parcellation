import pathlib
import os
from dotenv import load_dotenv
load_dotenv()
PYTHON3 = os.getenv("PYTHON3")
PATH_DICOM2NII = pathlib.Path(__file__).parent.joinpath('dicom2nii','main_call.py').absolute()
FSL_FLIRT = os.getenv("FSL_FLIRT")

# LOCAL_DB = pathlib.Path(__file__).parent.joinpath('database.sqlite3').absolute()
LOCAL_DB = os.getenv("LOCAL_DB","database.sqlite3")