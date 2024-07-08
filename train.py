import keras

from src.dataset import preprocess
from utils import prepare_log_folder
import argparse
from src.model import dice_coef, dice_coef_loss, build_model
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
import os
from tensorflow.keras.optimizers.legacy import Adam


def set_callbacks(log_dir):
    checkpointer = ModelCheckpoint(filepath=os.path.join(log_dir, 'checkpoint.keras'),
                                   verbose=1,
                                   save_best_only=True,
                                   monitor="dice_coef_loss")

    csv_logger = CSVLogger(os.path.join(log_dir, 'log.csv'), append=True, separator=';')
    callbacks = [checkpointer, csv_logger]
    return callbacks


def train(path: str, LR: float, loss: str, epochs: int, batch_size: int):
    keras.saving.get_custom_objects().clear()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess(path, 0.6)
    log_dir = prepare_log_folder()

    optimizer = Adam(learning_rate=LR)
    metrics = [dice_coef, dice_coef_loss, tf.keras.metrics.MeanIoU(num_classes=2)]

    unet_model = build_model()
    unet_model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)

    callbacks = set_callbacks(log_dir)
    history = unet_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                             verbose=1, validation_data=(X_val, Y_val),
                             callbacks=callbacks)

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
    train(args.dataset_path, args.LR, args.loss, args.epochs, args.batch_size)
