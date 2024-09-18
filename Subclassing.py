from CellMalaryaDetection.ReadData import read_image_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, MaxPooling2D, BatchNormalization, Input, Dropout, RandomFlip, RandomRotation, Resizing, Rescaling
from tensorflow.keras.layers import Layer
import tensorflow as tf
import matplotlib.pyplot as plt

path = "D:/0penThis/imageProcessing/archive/cell_images/cell_images/"
r = read_image_data()
CONFIGURATION = {
  "LEARNING_RATE": 0.001,
  "N_EPOCHS": 1,
  "BATCH_SIZE": 128,
  "DROPOUT_RATE": 0.0,
  "IM_SIZE": 50,
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

    self.dense_3 = Dense(2, activation="sigmoid")

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
lenet_sub_classed(tf.zeros([2, 50, 50, 3]))
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
lenet_sub_classed.compile(
              optimizer= Adam(
                  learning_rate = CONFIGURATION['LEARNING_RATE']),
              loss='binary_crossentropy',
              metrics=['accuracy'],
              )
history = lenet_sub_classed.fit(train_dataset,validation_data=val_dataset, epochs=10)
# test_data = tf.data.Dataset.range(8)
# test_data = list(normalized_dataset.as_numpy_iterator())
# test = lenet_model.evaluate(test_data)
# print(test)

# test_data = list(normalized_dataset.as_numpy_iterator())[:20]
test = lenet_sub_classed.evaluate(test_dataset)
print(test)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
