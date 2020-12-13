import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image_dataset_from_directory

MODELS = ['MobileNetV2']


class TransferLearning:
    IMG_HEIGHT = None
    IMG_WIDTH = None
    IMG_SIZE = None
    BATCH_SIZE = None
    DATASET = 'dataset/chest_xray'

    def __init__(self, IMG_HEIGHT=160, IMG_WIDTH=160):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, 3)

    def get_fine_tuning_model(self, model):
        fine_tuning_models = {'MobileNetV2': tf.keras.applications.MobileNetV2(input_shape=self.IMG_SIZE,
                                                                               include_top=False,
                                                                               weights='imagenet'),
                              }
        return fine_tuning_models.get(model)

    def prepare_dataset(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           validation_split=0.4)  # set validation split

        train_generator = train_datagen.flow_from_directory(
            self.DATASET,
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='binary',
            seed=123,
            subset='training')  # set as training data

        validation_generator = train_datagen.flow_from_directory(
            self.DATASET,  # same directory as training data
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='binary',
            seed=123,
            subset='validation')  # set as validation data
        return train_generator, validation_generator
