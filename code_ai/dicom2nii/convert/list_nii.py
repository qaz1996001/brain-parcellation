import argparse
import pathlib

import pandas as pd


def list_nifti(data_path: pathlib.Path):
    nifti_folder_list = []
    for study_path in list(data_path.iterdir()):
        series_path_list = list(study_path.rglob('*.nii.gz'))
        for series_path in series_path_list:
            print(study_path)
            nifti_folder_list.append([study_path.name, series_path.name, 1])
    df = pd.DataFrame(nifti_folder_list)
    df.columns = ['study_id', 'series_name', 'count']
    df['patients_id'] = df['study_id'].map(lambda x: x.split('_')[0])
    df['study_date'] = df['study_id'].map(lambda x: x.split('_')[1])
    df['accession_number'] = df['study_id'].map(lambda x: x.split('_')[-1])
    df1 = df.pivot_table(index=['patients_id',
                                'study_date',
                                'accession_number'
                                ], columns='series_name', aggfunc='count')
    df1.columns = df1.columns.get_level_values('series_name')
    df1 = df1.fillna(0)
    df1.to_excel(str(data_path.parent.joinpath(f'{data_path.name}_df_nifti.xlsx')), merge_cells=False)
    # df1.to_csv(str(data_path.parent.joinpath(f'{data_path.name}_df_nifti.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', '--input_nifti', dest='input_nifti', type=str, required=True,
                        help="input the nifti' folder.\r\n")
    args = parser.parse_args()
    data_path = pathlib.Path(args.input_nifti)
    list_nifti(data_path)
