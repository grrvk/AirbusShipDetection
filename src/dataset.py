import numpy as np
import cv2
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from utils import file_check


def enc_to_pixels(enc):
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
    w, h, _ = image.shape
    obj_map = np.zeros((w * h, 1), dtype=int)

    for i, (index, row) in enumerate(encodes.iterrows()):
        rle_mask_pixels = enc_to_pixels(row['EncodedPixels'])
        obj_map[rle_mask_pixels] = 255

    mask = np.reshape(obj_map, (w, h)).T
    return mask


def split_data(data, train_rate):
    assert (0 < train_rate <= 1)
    remaining_rate = (1 - train_rate) / 2
    image_names = data['ImageId'].drop_duplicates().tolist()
    train_image_names = image_names[:int(len(image_names) * train_rate)]
    val_image_names = image_names[
                      int(len(image_names) * train_rate):int(len(image_names) * (train_rate + remaining_rate))]
    test_image_names = image_names[int(len(image_names) * (train_rate + remaining_rate)):]
    return train_image_names, val_image_names, test_image_names


def get_dataset(data, names, directory: str):
    if len(names) == 0: return None, None
    images = []
    masks = []

    for i, image_name in enumerate(names):
        encodes = data[data['ImageId'] == image_name]
        image = cv2.imread(os.path.join(directory, image_name))
        mask = create_mask(image, encodes)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)


def prepare_masks(masks, le, classes=2):
    n, h, w = masks.shape
    masks_reshaped = masks.reshape(-1, 1)
    masks_labeled = le.fit_transform(masks_reshaped.ravel())
    masks_orig = masks_labeled.reshape(n, h, w)

    masks = np.expand_dims(masks_orig, axis=3)
    return to_categorical(masks, num_classes=classes)


def reshape(Y):
    if Y is None: return None
    le = LabelEncoder()
    Y = prepare_masks(Y, le)
    return Y


def preprocess(path, train_split_rate=0.6):
    train_folder_path, train_csv_path = file_check(path)
    data = pd.read_csv(train_csv_path).dropna()
    train_image_names, val_image_names, test_image_names = split_data(data, train_split_rate)
    X_train, Y_train = get_dataset(data, train_image_names, train_folder_path)
    X_val, Y_val = get_dataset(data, val_image_names, train_folder_path)
    X_test, Y_test = get_dataset(data, test_image_names, train_folder_path)
    print('Train dataset shape:\n'
          f'X: {X_train.shape}, Y: {Y_train.shape}\n'
          'Val dataset shape:\n'
          f'X: {X_val.shape}, Y: {Y_val.shape}\n')
    return X_train, reshape(Y_train), X_val, reshape(Y_val), X_test, reshape(Y_test)
