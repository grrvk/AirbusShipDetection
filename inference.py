import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from matplotlib import pyplot as plt
from glob import glob

from src.model import dice_coef, dice_coef_loss


def load_model(model_path='log/checkpoint.keras'):
    custom_objects = {"dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=True)


def inference(image_path, output_path='output'):
    image_name = image_path.split('/')[-1]
    os.makedirs(output_path, exist_ok=True)
    model = load_model()
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
    image = np.expand_dims(image, 0)
    prediction = np.argmax(model.predict(image), axis=3)[0, :, :]

    plt.title('Prediction')
    plt.axis('off')
    plt.imshow(prediction)
    prediction[prediction == 1] = 255
    cv2.imwrite(os.path.join(output_path, image_name), prediction)


def inference_dataset(path, output_path='output'):
    assert os.path.isdir(path)
    images = []
    for im_type in ['*.png', '*.jpg']: images.extend(glob(os.path.join(path, im_type)))
    for image in images:
        inference(image, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--path', metavar='str', type=str,
                        help='Path to the image or directory of images')

    args = parser.parse_args()
    if os.path.isdir(args.path):
        inference_dataset(args.path)
    elif os.path.isfile(args.path):
        inference(args.path)
    else:
        raise ValueError('Path should be a directory or a image')
