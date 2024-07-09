"""
Script to train the model
"""

from src.dataset import preprocess
from utils import prepare_log_folder
import argparse
from src.model import dice_coef, dice_coef_loss, build_model
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
import os
from tensorflow.keras.optimizers.legacy import Adam


def set_callbacks(log_dir):
    """
    Set callbacks
    :param log_dir: path to the directory to save log csv file
    :return: array of callbacks
    """
    checkpointer = ModelCheckpoint(filepath=os.path.join(log_dir, 'checkpoint.keras'),
                                   verbose=1,
                                   monitor="dice_coef_loss")

    csv_logger = CSVLogger(os.path.join(log_dir, 'log.csv'), append=True, separator=';')
    callbacks = [checkpointer, csv_logger]
    return callbacks


def train(path: str, LR: float, loss: str, epochs: int, batch_size: int):
    """
    Train and evaluate the model
    :param path: path to the dataset folder
    :param LR: learning rate value
    :param loss: model loss type
    :param epochs: number of epochs for training
    :param batch_size: batch size for training
    :return: ---
    """
    # get train, validation and test datasets
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess(path, 0.6)
    log_dir = prepare_log_folder()  # prepare directory to save model and log.csv

    optimizer = Adam(learning_rate=LR)  # set model optimizer
    metrics = [dice_coef, dice_coef_loss, tf.keras.metrics.MeanIoU(num_classes=2)]  # set model metrics

    # build and compile the model
    unet_model = build_model()
    unet_model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)

    # get callbacks array
    callbacks = set_callbacks(log_dir)

    # train and validate the model
    history = unet_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                             verbose=1, validation_data=(X_val, Y_val),
                             callbacks=callbacks) if X_val \
        else unet_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    # evaluate the model
    if X_test:
        unet_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        unet_model.evaluate(X_test, Y_test, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run train')
    parser.add_argument('--dataset_path', metavar='str', type=str,
                        help='General path to the dataset')
    parser.add_argument('--LR', metavar='N', type=float,
                        help='Learning rate')
    parser.add_argument('--loss', metavar='str', type=str,
                        help='Loss function')
    parser.add_argument('--epochs', metavar='N', type=int,
                        help='Number of epochs to run train')
    parser.add_argument('--batch_size', metavar='N', type=int,
                        help='Size of a batch')

    args = parser.parse_args()
    train(args.dataset_path, args.LR, args.loss, args.epochs, args.batch_size)  # run training
