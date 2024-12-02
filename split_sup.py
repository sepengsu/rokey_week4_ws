import os, random
import shutil

def list_files(directory,format='.jpg'):
    """
    List all files in the directory with the specified format
    :param directory: the directory to list files
    :param format: the format of the files
    :return: a list of files with the specified format
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(format):
            files.append(file)
    return files

def split_files(files,vaild_num):
    """
    Split files into training and validation sets
    :param files: a list of files
    :param ratio: the ratio of training set
    :return: a tuple of training and validation sets
    """
    random.seed(42)
    random.shuffle(files)
    return files[vaild_num:], files[:vaild_num] # return a tuple

def find_json_files(files):
    """
    Find json files in the list of files
    :param files: a list of files
    :return: a list of json files
    """
    json_files = []
    for file in files:
        filename = file+'.anno.json'
        json_files.append(filename)
    return json_files

def copy_files(files, origin_folder, target_folder):
    """
    Copy files from the origin folder to the target folder
    :param files: a list of files
    :param origin_folder: the origin folder
    :param target_folder: the target folder
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for file in files:
        src = os.path.join(origin_folder, file)
        dst = os.path.join(target_folder, file)
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            raise ValueError(src + ' is not file')

if __name__ == '__main__':
    origin_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\day4_raw'
    train_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\train_data\image'
    vaild_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\valid_data\image'
    train_json_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\train_data\label'
    vaild_json_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\valid_data\label'
    files = list_files(origin_folder)
    vaild_num = 89
    train_files, vaild_files = split_files(files,vaild_num)

    train_json_files = find_json_files(train_files)
    vaild_json_files = find_json_files(vaild_files)

    copy_files(train_files, origin_folder, train_folder)
    copy_files(vaild_files, origin_folder, vaild_folder)
    print('images Split files successfully!')
    copy_files(train_json_files, origin_folder, train_json_folder)
    copy_files(vaild_json_files, origin_folder, vaild_json_folder)
    print('json Split files successfully!')
