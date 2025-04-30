import pathlib
import os
from dotenv import load_dotenv
load_dotenv()

PYTHON3 = os.getenv("PYTHON3")
PATH_DICOM2NII = pathlib.Path(__file__).parent.joinpath('dicom2nii','main_call.py').absolute()
FSL_FLIRT = os.getenv("FSL_FLIRT")

print(os.getenv("FSL_FLIRT"))