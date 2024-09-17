from CellMalaryaDetection.ReadData import read_image_data

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, MaxPooling2D, BatchNormalization, Input, Dropout, RandomFlip, RandomRotation, Resizing, Rescaling
from tensorflow.keras.layers import Layer
import tensorflow as tf


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



class FeatureExtractor(Layer):
  def __init__(self, filters, kernel_size, strides, padding, activation, pool_size,):
    super(FeatureExtractor, self).__init__()

    self.conv_1 = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, activation = activation)
    self.batch_1 = BatchNormalization()
    self.pool_1 = MaxPool2D (pool_size = pool_size, strides= 2*strides)

    self.conv_2 = Conv2D(filters = filters*2, kernel_size = kernel_size, strides = strides, padding = padding, activation = activation)
    self.batch_2 = BatchNormalization()
    self.pool_2 = MaxPool2D (pool_size = pool_size, strides= 2*strides)

  def call(self, x):

    x = self.conv_1(x)
    x = self.batch_1(x)
    x = self.pool_1(x)

    x = self.conv_2(x)
    x = self.batch_2(x)
    x = self.pool_2(x)

    return x
# feature_sub_classed = FeatureExtractor(8, 3, 1, "valid", "relu", 2)


class LenetModel(tf.keras.models.Model):
  def __init__(self):
    super(LenetModel, self).__init__()

    self.feature_extractor = FeatureExtractor(8, 3, 1, "valid", "relu", 2)

    self.flatten = Flatten()

    self.dense_1 = Dense(100, activation="relu")
    self.batch_1 = BatchNormalization()

    self.dense_2 = Dense(10, activation="relu")
    self.batch_2 = BatchNormalization()

    self.dense_3 = Dense(1, activation="sigmoid")

  def call(self, x):
    x = self.feature_extractor(x)
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.batch_1(x)
    x = self.dense_2(x)
    x = self.batch_2(x)
    x = self.dense_3(x)

    return x


lenet_sub_classed = LenetModel()
lenet_sub_classed(tf.zeros([2, 224, 224, 3]))
lenet_sub_classed.summary()




# IM_SIZE = CONFIGURATION['IM_SIZE']
#
# func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image")
#
# x = feature_sub_classed(func_input)
#
# x = Flatten()(x)
#
# x = Dense(100, activation = "relu")(x)
# x = BatchNormalization()(x)
#
# x = Dense(10, activation = "relu")(x)
# x = BatchNormalization()(x)
#
# func_output = Dense(1, activation = "sigmoid")(x)
#
# lenet_model_func = tf.keras.models.Model(func_input, func_output, name = "Lenet_Model")
# lenet_model_func.summary()
lenet_sub_classed.compile(optimizer="adam", loss="mse")
# lenet_sub_classed.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = lenet_sub_classed.fit(normalized_dataset, epochs=1)