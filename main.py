import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from model.fine_tuning import TransferLearning, MODELS

a = TransferLearning()
model = a.get_fine_tuning_model(MODELS[0])
model.summary()
train, val = a.prepare_dataset()
print(train.class_indices)
print(sum(val.labels)/val.samples)
print(sum(train.labels)/train.samples)
