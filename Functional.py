from CellMalaryaDetection.ReadData import read_image_data
import tensorflow as tf### models
# from tensorflow.keras.models import Model
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
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, MaxPooling2D, BatchNormalization, Input, Dropout, RandomFlip, RandomRotation, Resizing, Rescaling
from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC, binary_accuracy
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2, L1
from tensorboard.plugins.hparams import api as hp

from preprocess import ttData


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
normalized_dataset = r.read(path,CONFIGURATION['IM_SIZE'], CONFIGURATION['IM_SIZE'], CONFIGURATION['BATCH_SIZE'])#48,48,8)




IM_SIZE = CONFIGURATION['IM_SIZE']
DROPOUT_RATE = CONFIGURATION['DROPOUT_RATE']
REGULARIZATION_RATE = CONFIGURATION['REGULARIZATION_RATE']
N_FILTERS = CONFIGURATION['N_FILTERS']
KERNEL_SIZE = CONFIGURATION['KERNEL_SIZE']
POOL_SIZE = CONFIGURATION['POOL_SIZE']
N_STRIDES = CONFIGURATION['N_STRIDES']
#lenet_model
func_input = Input(shape = (IM_SIZE,IM_SIZE,3), name = "Input Image")
x = Conv2D(32, (3, 3), activation='relu', input_shape=(IM_SIZE, IM_SIZE, 3))(func_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)#(feature_extracter_output)
x = Flatten()(x)
func_output = Dense(10)(x)
lenet_model = tf.keras.models.Model(func_input, func_output, name='Lenet_Model')
lenet_model.summary()



#feature extractor

func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image")

x = Conv2D(filters = 6, kernel_size = 3, strides=1, padding='valid', activation = 'relu')(func_input)
x = BatchNormalization()(x)
x = MaxPool2D (pool_size = 2, strides= 2)(x)

x = Conv2D(filters = 16, kernel_size = 3, strides=1, padding='valid', activation = 'relu')(x)
x = BatchNormalization()(x)
output = MaxPool2D (pool_size = 2, strides= 2)(x)

feature_extractor = tf.keras.models.Model(func_input, output, name = "Feature_Extractor")
feature_extractor.summary()




func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image")

x = feature_extractor(func_input)

x = Flatten()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(10, activation = "relu")(x)
x = BatchNormalization()(x)

func_output = Dense(2, activation = "sigmoid")(x)

lenet_model_func = tf.keras.models.Model(func_input, func_output, name = "Lenet_Model")
lenet_model_func.summary()


lenet_model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = lenet_model.fit(normalized_dataset, epochs=1)