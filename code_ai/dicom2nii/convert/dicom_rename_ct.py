from concurrent.futures import ThreadPoolExecutor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help="input the raw dicom folder.\r\n")
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                        help="output the rename dicom folder.\r\n"
                             "Example: python rename_folder.py -i input_path -o output_path")
    parser.add_argument('--work', dest='work', type=int, default=4,
                        help="Thread count.\r\n"
                             "Example: python rename_folder.py -i input_path -o output_path --work 8")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    output_path = args.output
    input_path = args.input
    work = min(8, max(1, args.work))
    executor = ThreadPoolExecutor(max_workers=work)
    with executor:
        convert_manager = ConvertManager(input_path=input_path, output_path=output_path)
        convert_manager.run(executor=executor)
