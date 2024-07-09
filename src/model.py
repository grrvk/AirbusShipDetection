"""
Script to build a simple U-Net model
"""

import tensorflow as tf
import keras
from keras.layers import *
from tensorflow.keras import backend as K


@keras.saving.register_keras_serializable(name='dice_coef')
def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Dice coefficient
    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :param smooth: value to avoid division by zero
    :return: dice coefficient
    """
    y_true_f = K.flatten(y_true)  # make ground truth matrix 1d array
    y_pred_f = K.flatten(y_pred)  # make ground truth matrix 1d array
    intersection = K.sum(y_true_f * y_pred_f)  # calculate intersection of truth and prediction
    union = K.sum(y_true_f) + K.sum(y_pred_f)  # calculate union of truth and prediction
    return (2. * intersection + smooth) / (union + smooth)


@keras.saving.register_keras_serializable(name='dice_coef_loss')
def dice_coef_loss(y_true, y_pred):
    """
    Dice coefficient loss
    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :return: dice coefficient loss
    """
    return 1 - dice_coef(y_true, y_pred)


def double_conv(x, n_filters):
    """
    Double convolution layer
    :param x: tensor inputs
    :param n_filters: number of filters for the convolution layer
    :return: tensor outputs
    """
    x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def downsample(x, n_filters):
    """
    Downsample layer
    :param x: tensor inputs
    :param n_filters: number of filters for the convolution layer
    :return: tensor outputs
    """
    f = double_conv(x, n_filters)
    p = MaxPool2D(2)(f)
    return f, p


def upsample(x, conv_features, n_filters):
    """
    Upsample layer
    :param x: tensor inputs from previous layer
    :param conv_features: tensor inputs from previous skip layer
    :param n_filters: number of filters for the convolution layer
    :return: tensor outputs
    """
    x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = concatenate([x, conv_features])
    x = double_conv(x, n_filters)
    return x


def build_model(img_size=(128, 128, 3), num_classes=2):
    """
    Model structure build
    :param img_size: image size for the input layer
    :param num_classes: number of classes to predict
    :return: model
    """
    # Input layer
    inputs = Input(shape=img_size)

    # Four downsample layers
    f1, p1 = downsample(inputs, 64)
    f2, p2 = downsample(p1, 128)
    f3, p3 = downsample(p2, 256)
    f4, p4 = downsample(p3, 512)

    # bottleneck of the model
    bottleneck = double_conv(p4, 1024)

    # Four upsample layers
    u6 = upsample(bottleneck, f4, 512)
    u7 = upsample(u6, f3, 256)
    u8 = upsample(u7, f2, 128)
    u9 = upsample(u8, f1, 64)

    # Output layer
    outputs = Conv2D(num_classes, 1, padding="same", activation='sigmoid')(u9)
    # Model
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model
