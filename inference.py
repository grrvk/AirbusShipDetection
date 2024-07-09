import pandas as pd
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
    custom_objects = {"dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=True)


def get_prediction(image_path, model):
    image = cv2.imread(image_path)
    w, h, _ = image.shape
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
    image = np.expand_dims(image, 0)
    prediction_mask = model.predict(image)
    prediction = np.argmax(prediction_mask, axis=3)[0, :, :]
    prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
    return prediction


def get_encoded_pixels(truth, seg_label):
    filtered_matrix = truth.copy()
    filtered_matrix[filtered_matrix != seg_label] = 0
    filtered_matrix = filtered_matrix.flatten(order='F')
    indices = list(itertools.filterfalse(lambda j: (filtered_matrix[j]-filtered_matrix[j-1] == 0),
                                         range(len(filtered_matrix))))
    enc_pixels = []
    for i in range(0, len(indices)):
        enc_pixels.append(indices[i]) if i % 2 == 0 else enc_pixels.append(indices[i]-indices[i-1])
    return ' '.join(str(x) for x in enc_pixels)


def inference(image_path, csvwriter, save_mask, model=None, model_path=None):
    save_dir = 'output'
    if save_mask: os.makedirs(save_dir, exist_ok=True)
    if model is None: model = load_model(model_path)

    image_name = image_path.split('/')[-1]
    prediction = get_prediction(image_path, model)
    labeled_array, _ = label(prediction)
    unique_segments = np.delete(np.unique(labeled_array), 0, 0)
    for i, seg_label in enumerate(unique_segments):
        enc_pixels = get_encoded_pixels(labeled_array, seg_label)
        csvwriter.writerow([image_name, enc_pixels])

    if save_mask: save_image(prediction, os.path.join(save_dir, image_name))


def inference_dataset(path, model_path, csvwriter, save_mask):
    assert os.path.isdir(path)
    model = load_model(model_path)
    images = []
    for im_type in ['*.png', '*.jpg']: images.extend(glob(os.path.join(path, im_type)))
    for image in images:
        inference(image, csvwriter, model, save_mask)


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
    csv_full_path = os.path.join(args.csv_path, 'result.csv')
    fields = ['ImageId', 'EncodedPixels']
    with open(csv_full_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        if os.path.isdir(args.inf_path):
            inference_dataset(args.inf_path, args.model_path, csvwriter, args.save_masks)
        elif os.path.isfile(args.inf_path):
            inference(args.inf_path, csvwriter, model_path=args.model_path, save_mask=args.save_masks)
        else:
            raise ValueError('Path should be a directory or a image')

