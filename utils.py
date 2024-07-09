"""
Additional small scripts to help working with files
"""

import os
from glob import glob
import cv2
from matplotlib import pyplot as plt


def file_check(path, tr_name='train_v2'):
    """
    Check do image folder and csv file exist
    :param path: path to the dataset directory
    :param tr_name: folder with images name
    :return: paths to the image folder and csv data
    """
    assert os.path.isdir(path) # check is path to the dataset directory valid
    # get full path to the folder with images for training
    train_folder_path = os.path.join(path, tr_name)
    # get full path to the csv file with data for training
    # if several - choose the first
    train_csv_path = [path for path in glob(os.path.join(path, '*.csv')) if 'train' in path][0]

    assert os.path.isdir(train_folder_path)  # validate path to the image folder
    assert os.path.isfile(train_csv_path)

    return train_folder_path, train_csv_path


def prepare_log_folder(log_dir='log'):
    """
    Check does folder to save log exist
    :param log_dir: path to the directory to save log
    :return: path
    """
    os.makedirs(log_dir, exist_ok=True)  # create directory
    return log_dir


def save_image(image, path):
    """
    Save mask if flag is True
    :param image: image to save
    :param path: path to save image
    :return: ---
    """
    plt.axis('off')
    plt.imshow(image)
    image[image == 1] = 255
    cv2.imwrite(path, image)