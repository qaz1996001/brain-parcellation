import argparse
import pathlib
from pydicom import dcmread, FileDataset, Dataset, DataElement
import pandas as pd
import orjson


def get_dicom_data(dicom_ds: FileDataset):
    # (0010,0020)	Patient ID	05470905
    # (0010,0040)	Patient Sex	M
    # (0010,0030)	Patient Birth Date	19481009
    # (0008,0020)	Study Date	20220616
    # (0008,0050)	Accession Number	21106150005
    patients_id = dicom_ds[0x10, 0x20].value
    gender = dicom_ds[0x10, 0x40].value
    birth_date = dicom_ds[0x10, 0x30].value
    studies_date = dicom_ds[0x08, 0x20].value
    studies_time = dicom_ds[0x08, 0x30].value
    studies_description = dicom_ds[0x08, 0x1030].value
    accession_number = dicom_ds[0x08, 0x50].value
    # (0008,0021)	Series Date	20220616
    series_date = dicom_ds[0x08, 0x21].value
    series_time = dicom_ds[0x08, 0x31].value
    if '.' in studies_time:
        studies_time = studies_time.split('.')[0]
    if '.' in series_time:
        series_time = series_time.split('.')[0]

    modality = dicom_ds[0x08, 0x60].value
    return dict(patient_id=patients_id, gender=gender, birth_date=birth_date,
                study_date=studies_date, study_time=studies_time,
                study_description=studies_description, accession_number=accession_number,
                series_date=series_date, series_time=series_time, modality=modality)


def list_dicom_study(data_path: pathlib.Path) -> pd.DataFrame:
    dicom_folder_list = []
    for study_path in list(data_path.iterdir()):
        series_folder_path_list = list(study_path.iterdir())
        for series_path in series_folder_path_list:
            if series_path.name == '.meta':
                continue
            # print(series_path)
            dicom_folder_list.append([study_path.name, series_path.name, 1])
    df = pd.DataFrame(dicom_folder_list)
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
    return df1


def list_dicom_series(data_path: pathlib.Path) -> pd.DataFrame:
    series_meta_list = []
    for study_path in data_path.iterdir():
        meta_folder_path = filter(lambda x: x.name == '.meta' ,study_path.iterdir())
        for meta_path in next(meta_folder_path).iterdir():
            print(meta_path)
            with open(f'{meta_path}', encoding='utf8') as file:
                orjson_dict = orjson.loads(file.read())
                for key, value in orjson_dict.items():
                    dicom_header  = Dataset.from_json(value[0])
                    data_dict = get_dicom_data(dicom_ds=dicom_header)
                    data_dict['series_description'] = meta_path.stem
                    series_meta_list.append(data_dict)
                    # break
    df = pd.DataFrame(series_meta_list)
    return df


def list_dicom(data_path: pathlib.Path):
    df1 = list_dicom_study(data_path=data_path)
    df2 = list_dicom_series(data_path=data_path)
    df1.to_excel(str(data_path.parent.joinpath(f'{data_path.name}_df_dicom.xlsx')),merge_cells=False,index=False)
    df1.to_csv(str(data_path.parent.joinpath(f'{data_path.name}_df_dicom.csv')),index=False)
    df2.to_excel(str(data_path.parent.joinpath(f'{data_path.name}_df_series.xlsx')), merge_cells=False,index=False)
    df2.to_csv(str(data_path.parent.joinpath(f'{data_path.name}_df_series.csv')),index=False)


__all__ = ['list_dicom', 'list_dicom_study', 'list_dicom_series']

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input', '--input_dicom', dest='input_dicom', type=str, #required=True,
#                         help="input the rename dicom folder.\r\n",
#                         default=r'D:\00_Chen\Task08\data\test_list'
#                         )
#     args = parser.parse_args()
#     data_path = pathlib.Path(args.input_dicom)
#     list_dicom(data_path)
