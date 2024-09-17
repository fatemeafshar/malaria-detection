import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
from readData import read_data
import pathlib


class read_image_data:
  def __init__(self):
    pass

  def read(self, path, image_width, image_height, batch_size, number_of_channels=3):
    color_mode = "rgb"
    if number_of_channels == 1:
      color_mode = "grayscale"
    elif number_of_channels == 4:
      color_mode = "rgba"
    data_dir = pathlib.Path(path).with_suffix('')
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      label_mode = "categorical",#int
      color_mode = color_mode,
      image_size=(image_width, image_height),
      batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)

    for image_batch, labels_batch in train_ds:
      print(image_batch.shape)
      print(labels_batch.shape)
      break
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))
    return normalized_ds


'''
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = "D:/0penThis/imageProcessing/archive/cell_images/cell_images/"#tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(48, 48),
  batch_size=4)


class_names = train_ds.class_names
print(class_names)



for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
normalization_layer = tf.keras.layers.Rescaling(1./255)


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

'''
# PIL.Image.open(str(roses[0]))