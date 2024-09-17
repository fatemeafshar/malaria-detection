from CellMalaryaDetection.ReadData import read_image_data
import tensorflow as tf### models
# from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC, binary_accuracy
import numpy as np### math computations
import matplotlib.pyplot as plt### plots
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime
import io
import os
import random
from PIL import Image
# import albumentations as A
import tensorflow_datasets as tfds
# import tensorflow_probability as tfp
# from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Input, Dropout, RandomFlip, RandomRotation, Resizing, Rescaling
from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC, binary_accuracy
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2, L1
from tensorboard.plugins.hparams import api as hp

from preprocess import ttData

path_train = 'D:/0penThis/imageProcessing/archive/cell_images/cell_images/'

# d = ttData()
# numberOfLabels = 2
# dataset, target = d.getData(path_train, numberOfLabels)
#
# print(dataset.shape)


# dataset, dataset_info = tfds.load('malaria', with_info=True,
#                                   as_supervised=True,
#                                   shuffle_files = True,
#                                   split=['train'])

path = "D:/0penThis/imageProcessing/archive/cell_images/cell_images/"
r = read_image_data()
CONFIGURATION = {
  "LEARNING_RATE": 0.001,
  "N_EPOCHS": 1,
  "BATCH_SIZE": 128,
  "DROPOUT_RATE": 0.0,
  "IM_SIZE": 224,
  "REGULARIZATION_RATE": 0.0,
  "N_FILTERS": 6,
  "KERNEL_SIZE": 3,
  "N_STRIDES": 1,
  "POOL_SIZE": 2,
  "N_DENSE_1": 100,
  "N_DENSE_2": 10,
}
full_dataset = r.read(path,CONFIGURATION['IM_SIZE'], CONFIGURATION['IM_SIZE'], CONFIGURATION['BATCH_SIZE'])#48,48,8)

DATASET_SIZE = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

full_dataset = full_dataset.shuffle(buffer_size = 10)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)



# for element in normalized_dataset:
#   print(element)


# model = tf.keras.Sequential([
#                              InputLayer(input_shape = ( 48, 48, 3)),
#
#                              Dense(128, activation = "relu"),
#                              Dense(128, activation = "relu"),
#                              Dense(128, activation = "relu"),
#                              Dense(1),
# ])
# model.summary()
# dot_img_file = '/tmp/model_1.png'
# # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
#
#
# model.compile(optimizer = Adam(learning_rate = 0.1), loss = MeanAbsoluteError())
#
# history = model.fit(normalized_dataset, epochs = 10, verbose = 1)



IM_SIZE = CONFIGURATION['IM_SIZE']
DROPOUT_RATE = CONFIGURATION['DROPOUT_RATE']
REGULARIZATION_RATE = CONFIGURATION['REGULARIZATION_RATE']
N_FILTERS = CONFIGURATION['N_FILTERS']
KERNEL_SIZE = CONFIGURATION['KERNEL_SIZE']
POOL_SIZE = CONFIGURATION['POOL_SIZE']
N_STRIDES = CONFIGURATION['N_STRIDES']

lenet_model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),

    Conv2D(filters=N_FILTERS, kernel_size=KERNEL_SIZE, strides=N_STRIDES, padding='valid',
           activation='relu', kernel_regularizer=L2(REGULARIZATION_RATE)),
    BatchNormalization(),
    MaxPool2D(pool_size=POOL_SIZE, strides=N_STRIDES * 2),
    Dropout(rate=DROPOUT_RATE),

    Conv2D(filters=N_FILTERS * 2 + 4, kernel_size=KERNEL_SIZE, strides=N_STRIDES, padding='valid',
           activation='relu', kernel_regularizer=L2(REGULARIZATION_RATE)),
    BatchNormalization(),
    MaxPool2D(pool_size=POOL_SIZE, strides=N_STRIDES * 2),

    Flatten(),

    Dense(CONFIGURATION['N_DENSE_1'], activation="relu", kernel_regularizer=L2(REGULARIZATION_RATE)),
    BatchNormalization(),
    Dropout(rate=DROPOUT_RATE),

    Dense(CONFIGURATION['N_DENSE_2'], activation="relu", kernel_regularizer=L2(REGULARIZATION_RATE)),
    BatchNormalization(),

    Dense(2, activation="sigmoid"),

])

lenet_model.summary()

lenet_model.compile(
              optimizer= Adam(
                  learning_rate = CONFIGURATION['LEARNING_RATE']),
              loss='binary_crossentropy',
              metrics=['accuracy'],
              )
history = lenet_model.fit(full_dataset,validation_data=val_dataset, epochs=1)
# test_data = tf.data.Dataset.range(8)
# test_data = list(normalized_dataset.as_numpy_iterator())
# test = lenet_model.evaluate(test_data)
# print(test)

# test_data = list(normalized_dataset.as_numpy_iterator())[:20]
test = lenet_model.evaluate(test_dataset)
print(test)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()





predict = lenet_model.predict(test_dataset)
labels = np.concatenate([y for x, y in test_dataset], axis=0)
images = np.concatenate([x for x, y in test_dataset], axis=0)

y_true = list(labels[:,0].numpy())
y_pred = list(lenet_model.predict(images)[:,0])

ind = np.arange(100)
plt.figure(figsize=(40,20))

width = 0.1

plt.bar(ind, y_pred, width, label='Predicted Car Price')
plt.bar(ind + width, y_true, width, label='Actual Car Price')

plt.xlabel('Actual vs Predicted Prices')
plt.ylabel('Car Price Prices')

plt.show()




# def test_model(history):
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'])
#     plt.show()
#
#     model.evaluate(X_test,y_test)
#
#
#
#     model.predict(tf.expand_dims(X_test[0], axis = 0 ))
#     y_true = list(y_test[:,0].numpy())
#     y_pred = list(model.predict(X_test)[:,0])
#
#     ind = np.arange(100)
#     plt.figure(figsize=(40,20))
#
#     width = 0.1
#
#     plt.bar(ind, y_pred, width, label='Predicted Car Price')
#     plt.bar(ind + width, y_true, width, label='Actual Car Price')
#
#     plt.xlabel('Actual vs Predicted Prices')
#     plt.ylabel('Car Price Prices')
#
#     plt.show()



