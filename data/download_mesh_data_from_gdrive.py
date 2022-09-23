import gdown
import shutil
import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('download_mesh', add_help=False)
    parser.add_argument(
        '--output_folder',
        dest='output_folder',
        type=str,
        default=os.getcwd() + '/meshes',
        help="""path for saving mesh files."""
        )
    return parser

def download_mesh_files(file_list, destination_directory):
    '''
    download mesh files from google drive
    inputs:
        file_list is a list, where each element contains the file name and the google drive file id
        destination_directory is where we want to save the downloaded file
    '''
    current_directory = os.getcwd()
    for item in file_list:
        file_name = item[0] + '.obj'
        file_id = item[1]
        url = 'https://drive.google.com/uc?id=' + file_id
        gdown.download(url, file_name, quiet=False)
        
        current_file_path = os.path.join(current_directory, file_name)
        output_file_path = os.path.join(destination_directory, file_name)
        shutil.move(current_file_path, output_file_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('TopoGrapher', parents=[get_args_parser()])
    args = parser.parse_args()
    destination_directory = args.output_folder
    
    file_list = [['hand1', '1fdYX1CFYg9qdRAYV7pttRlEv74QyRqhZ'],
                 ['moomoo_s0', '1z4whYv39rHoSk9BoTUAHRsM2Mci-LW-i']]
    download_mesh_files(file_list, destination_directory)




