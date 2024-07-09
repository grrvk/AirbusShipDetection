"""
Script to inference pretrained model on image or dataset
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
import csv
from glob import glob
import itertools
from scipy.ndimage import label
from src.model import dice_coef, dice_coef_loss
from utils import save_image


def load_model(model_path):
    """
    Load pretrained model
    :param model_path: path to the trained model checkpoint
    :return: keras model instance
    """
    custom_objects = {"dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss}  # set custom metrics
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=True)


def get_prediction(image_path, model):
    """
    get model prediction from image
    :param image_path: path to the image to be predicted
    :param model: keras model instance
    :return: prediction mask
    """
    # read and prepare image to pass to the model
    image = cv2.imread(image_path)
    w, h, _ = image.shape
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
    image = np.expand_dims(image, 0)

    # predict mask of objects
    prediction_mask = model.predict(image)

    # select predicted (value 1) pixels from two channels of prediction mask
    prediction = np.argmax(prediction_mask, axis=3)[0, :, :]
    # resize to the actual image size
    prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
    return prediction


def get_encoded_pixels(labeled_mask, seg_label):
    """
    Get encoded pixels of segment with given label
    :param labeled_mask: labeled mask as numpy matrix
    :param seg_label: label of the segment
    :return: string of encoded pixels
    """
    # copy labeled matrix, remove all segments with other labels
    filtered_matrix = labeled_mask.copy()
    filtered_matrix[filtered_matrix != seg_label] = 0

    # flatten matrix by columns to get 1d array
    filtered_matrix = filtered_matrix.flatten(order='F')

    # get indices of all non zero values in array
    indices = list(itertools.filterfalse(lambda j: (filtered_matrix[j]-filtered_matrix[j-1] == 0),
                                         range(len(filtered_matrix))))
    # Get the beginning value and count of consecutive values for each consecutive subarray
    enc_pixels = []
    for i in range(0, len(indices)):
        enc_pixels.append(indices[i]) if i % 2 == 0 else enc_pixels.append(indices[i]-indices[i-1])
    return ' '.join(str(x) for x in enc_pixels)


def inference(image_path, csvwriter, save_mask: bool, model=None, model_path=None):
    """
    Run inference on one image
    :param image_path: path to the image to be predicted
    :param csvwriter: csv writer instance
    :param save_mask: flag to save masks as images
    :param model: keras model instance
    :param model_path: path to load model from if no model given
    :return: ---
    """
    # set directory to save mask images to if flag is True, check directory existence
    save_dir = 'output'
    if save_mask: os.makedirs(save_dir, exist_ok=True)

    if model is None: model = load_model(model_path)  # load model if none given

    image_name = image_path.split('/')[-1]  # get image name
    prediction = get_prediction(image_path, model)  # get prediction mask
    labeled_array, _ = label(prediction)  # label each non zero prediction
    unique_segments = np.delete(np.unique(labeled_array), 0, 0)  # get number of instances

    # for each labeled instance get and write to csv encoded pixels
    for i, seg_label in enumerate(unique_segments):
        enc_pixels = get_encoded_pixels(labeled_array, seg_label)
        csvwriter.writerow([image_name, enc_pixels])

    # save prediction mask as image if flag is True
    if save_mask: save_image(prediction, os.path.join(save_dir, image_name))


def inference_dataset(path, model_path, csvwriter, save_mask: bool):
    """
    Run inference on dataset of several images
    :param path: path to the dataset of images
    :param model_path: path to load model from
    :param csvwriter: csv writer instance
    :param save_mask: flag to save masks as images
    :return: ---
    """
    model = load_model(model_path)  # load model

    # get images and run inference on each
    images = []
    for im_type in ['*.png', '*.jpg']: images.extend(glob(os.path.join(path, im_type)))
    for image in images:
        inference(image, csvwriter, model=model, save_mask=save_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--inf_path', metavar='str', type=str,
                        help='Path to the image or directory of images')
    parser.add_argument('--model_path', metavar='str', type=str,
                        help='Path to the model checkpoint', default='log/checkpoint.keras')
    parser.add_argument('--csv_path', metavar='str', type=str,
                        help='Path to where to save .csv', default='.')
    parser.add_argument('--save_masks', action='store_true', default=False)

    args = parser.parse_args()
    csv_full_path = os.path.join(args.csv_path, 'result.csv')  # get full path to csv
    columns = ['ImageId', 'EncodedPixels']  # initialize column names

    # run inference and write into csv file
    with open(csv_full_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(columns)  # write header
        if os.path.isdir(args.inf_path):
            # run if object from path is a directory
            inference_dataset(args.inf_path, args.model_path, csvwriter, args.save_masks)
        elif os.path.isfile(args.inf_path):
            # run if object from path is a file
            inference(args.inf_path, csvwriter, model_path=args.model_path, save_mask=args.save_masks)
        else:
            # else raise error
            raise ValueError('Path should be a directory or a image')

