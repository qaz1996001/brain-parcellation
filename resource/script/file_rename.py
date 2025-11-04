import pathlib

if __name__ == '__main__':
    data_path = pathlib.Path(__file__).parent.parent.joinpath('sql')
    print('data_path',data_path)
    file_path_list = sorted(data_path.iterdir())
    print(file_path_list)
    for file_path in file_path_list:
        if file_path.name.startswith('omi_'):
            re_name = file_path.parent.joinpath(file_path.name.replace('omi_', ''))
            file_path.rename(re_name)
            print(re_name)
