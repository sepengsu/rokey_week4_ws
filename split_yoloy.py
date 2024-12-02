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

def split_files(files,vaild_num,test_num):
    """
    Split files into training and validation sets
    :param files: a list of files
    :param ratio: the ratio of training set
    :return: a tuple of training and validation, test sets
    순서는 validation, test, train
    """
    random.seed(42)
    random.shuffle(files)
    #
    return files[:vaild_num], files[vaild_num:vaild_num+test_num], files[vaild_num+test_num:]

def find_txt_files(files):
    """
    Find json files in the list of files
    :param files: a list of files
    :return: a list of json files
    """
    json_files = []
    for file in files:
        filename = file.split('.')[0]+'.txt'
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
    origin_image_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\dfive_final\images'
    origin_txt_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\dfive_final\labels'
    train_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\project\data\images\train'
    vaild_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\project\data\images\val'
    
    train_txt_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\project\data\labels\train'
    vaild_txt_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\project\data\labels\val'

    test_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\project\data\images\test'
    test_txt_folder = r'C:\Users\na062\Desktop\rokey_week4_ws\project\data\labels\test'

    files = list_files(origin_image_folder)
    vaild_num = 100
    test_num = 43
    vaild_files, test_files, train_files = split_files(files, vaild_num, test_num)

    train_txt_files = find_txt_files(train_files)
    vaild_txt_files = find_txt_files(vaild_files)
    test_txt_files = find_txt_files(test_files)

    copy_files(train_files, origin_image_folder, train_folder)
    copy_files(vaild_files, origin_image_folder, vaild_folder)
    copy_files(test_files, origin_image_folder, test_folder)
    print('Images Split files successfully!')
    copy_files(train_txt_files, origin_txt_folder, train_txt_folder)
    copy_files(vaild_txt_files, origin_txt_folder, vaild_txt_folder)
    copy_files(test_txt_files, origin_txt_folder, test_txt_folder)
    print('labels Split files successfully!')
