import argparse
import pathlib
from code_ai.pipeline.dicomseg import cmb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        help="用於輸入的檔案",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="用於輸出結果的資料夾",
    )

