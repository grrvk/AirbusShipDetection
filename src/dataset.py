"""
Script to create masks for images and prepare dataset for training.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from utils import file_check


def enc_to_pixels(enc):
    """
    Calculate actual pixel values in flattened image
    :param enc: string of encoded pixels from train csv
    :return: array of pixel values in flattened image
    """
    # string to integer array
    enc = list(map(int, enc.split(' ')))

    pixel = []
    pixel_count = []
    # separate starting pixels and their range
    for i in range(0, len(enc)):
        pixel.append(enc[i]) if i % 2 == 0 else pixel_count.append(enc[i])

    # calculate actual pixel values
    true_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i] - 1)) for i in range(0, len(pixel))]
    result_pixels = sum(true_pixels, [])
    return result_pixels


def create_mask(image, encodes):
    """
    Create mask from encoded pixels
    :param image: image as numpy array
    :param encodes: dataframe containing encoded pixels for image
    :return: mask as numpy array
    """
    # create 1d blank mask matrix
    w, h, _ = image.shape
    obj_map = np.zeros((w * h, 1), dtype=int)

    # mark corresponding to encoded pixels objects on mask
    for i, (index, row) in enumerate(encodes.iterrows()):
        true_pixels = enc_to_pixels(row['EncodedPixels'])
        obj_map[true_pixels] = 255

    # reshape mask matrix to 2d
    mask = np.reshape(obj_map, (w, h)).T
    plt.imshow(mask)
    plt.axis('off')
    plt.show()
    return mask


def split_data(data, train_rate):
    """
    Split data into train, validation and test sets by image ids
    :param data: csv data dataframe
    :param train_rate: split rate for train data
    :return: lists of train, validation and test image names
    """
    # check is rate valid, calculate val and test rates
    assert (0 < train_rate <= 1)
    remaining_rate = (1 - train_rate) / 2

    # split data into train, val, test according to rates
    image_names = data['ImageId'].drop_duplicates().tolist()
    train_image_names = image_names[:int(len(image_names) * train_rate)]
    val_image_names = image_names[int(len(image_names) * train_rate):int(len(image_names) * (train_rate + remaining_rate))]
    test_image_names = image_names[int(len(image_names) * (train_rate + remaining_rate)):]
    return train_image_names, val_image_names, test_image_names


def get_dataset(data, names, directory: str):
    """
    Create numpy arrays of images and masks
    :param data: csv data dataframe
    :param names: image names to create masks for
    :param directory: path to images directory
    :return: numpy arrays of images and masks
    """
    # if due to split there is no data - return None
    if len(names) == 0: return None, None
    images = []
    masks = []

    # create masks for images
    for i, image_name in enumerate(names):
        encodes = data[data['ImageId'] == image_name]
        image = cv2.imread(os.path.join(directory, image_name))
        mask = create_mask(image, encodes)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)


def reshape(masks, classes=2):
    """
    :param masks: numpy array of masks
    :param classes: number of classes (class types + background)
    :return: reshaped and labeled masks
    """
    if masks is None: return None
    le = LabelEncoder()
    n, h, w = masks.shape
    masks_reshaped = masks.reshape(-1, 1)
    masks_labeled = le.fit_transform(masks_reshaped.ravel())
    masks_orig = masks_labeled.reshape(n, h, w)
    masks = np.expand_dims(masks_orig, axis=3)
    return to_categorical(masks, num_classes=classes)


def preprocess(path, train_split_rate=0.8):
    """
    Create datasets for training, validation and testing
    :param path: path to csv with data
    :param train_split_rate: rate to split data between train and else
    :return: X, Y (images and masks) for training, validation and testing
    """
    # check are images directory and csv file paths are valid and existing
    train_folder_path, train_csv_path = file_check(path)

    # read csv into pandas dataframe
    data = pd.read_csv(train_csv_path).dropna()

    # split data into train, validation and test
    train_image_names, val_image_names, test_image_names = split_data(data, train_split_rate)

    # create datasets
    X_train, Y_train = get_dataset(data, train_image_names, train_folder_path)
    X_val, Y_val = get_dataset(data, val_image_names, train_folder_path)
    X_test, Y_test = get_dataset(data, test_image_names, train_folder_path)
    return X_train, reshape(Y_train), X_val, reshape(Y_val), X_test, reshape(Y_test)
