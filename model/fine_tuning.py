import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

MODELS = {'VGG16': (224, 224),
          'VGG19': (224, 224),
          'InceptionV3': (299, 299),
          'Xception': (299, 299),
          'DenseNet201': (224, 224),
          'MobileNetV2': (224, 224),
          'InceptionResNetV2': (299, 299),
          'ResNet50': (224, 224),
          'ResNet50V2': (224, 224),
          }


def plot_history(history, model, initial_epochs=None, is_fine_tuning=False):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    if not is_fine_tuning:
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy ' + model)

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig('figures/' + model + '.png')
    else:
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([initial_epochs - 1, initial_epochs - 1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([initial_epochs - 1, initial_epochs - 1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig('figures/' + model + '.png')


def plot_cm(model, labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
    plt.savefig('figures/' + 'confusion_matrix_' + model + '.png')


class TransferLearning:
    IMG_HEIGHT = None
    IMG_WIDTH = None
    IMG_SIZE = None
    BATCH_SIZE = 32
    DATASET = 'dataset/chest_xray'

    def __init__(self):
        self.IMG_HEIGHT = None
        self.IMG_WIDTH = None
        self.IMG_SIZE = None

    def get_fine_tuning_model(self, model):
        shape = MODELS.get(model)
        self.IMG_HEIGHT = shape[0]
        self.IMG_WIDTH = shape[1]
        self.IMG_SIZE = shape + (3,)
        fine_tuning_models = {'VGG16': tf.keras.applications.VGG16(input_shape=self.IMG_SIZE,
                                                                   include_top=False,
                                                                   weights='imagenet'),
                              'VGG19': tf.keras.applications.VGG19(input_shape=self.IMG_SIZE,
                                                                   include_top=False,
                                                                   weights='imagenet'),
                              'InceptionV3': tf.keras.applications.InceptionV3(input_shape=self.IMG_SIZE,
                                                                               include_top=False,
                                                                               weights='imagenet'),
                              'Xception': tf.keras.applications.Xception(input_shape=self.IMG_SIZE,
                                                                         include_top=False,
                                                                         weights='imagenet'),
                              'DenseNet201': tf.keras.applications.DenseNet201(input_shape=self.IMG_SIZE,
                                                                               include_top=False,
                                                                               weights='imagenet'),
                              'MobileNetV2': tf.keras.applications.MobileNetV2(input_shape=self.IMG_SIZE,
                                                                               include_top=False,
                                                                               weights='imagenet'),
                              'InceptionResNetV2': tf.keras.applications.InceptionResNetV2(input_shape=self.IMG_SIZE,
                                                                                           include_top=False,
                                                                                           weights='imagenet'),
                              'ResNet50': tf.keras.applications.ResNet50(input_shape=self.IMG_SIZE,
                                                                         include_top=False,
                                                                         weights='imagenet'),
                              'ResNet50V2': tf.keras.applications.ResNet50V2(input_shape=self.IMG_SIZE,
                                                                             include_top=False,
                                                                             weights='imagenet'),
                              }
        return fine_tuning_models.get(model), self.IMG_SIZE

    def prepare_dataset(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=24,
                                           width_shift_range=0.15,
                                           height_shift_range=0.2,
                                           fill_mode='constant',
                                           shear_range=16,

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
