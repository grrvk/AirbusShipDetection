import os
from glob import glob


def file_check(path, tr_name='train_v2'):
    train_folder_path = os.path.join(path, tr_name)
    train_csv_path = [path for path in glob(os.path.join(path, '*.csv')) if 'train' in path][0]
    print(train_csv_path)
    assert os.path.isdir(train_folder_path)
    assert os.path.isfile(train_csv_path)

    return train_folder_path, train_csv_path


def prepare_log_folder(log_dir='log'):
    os.makedirs(log_dir, exist_ok=True)
    return log_dir