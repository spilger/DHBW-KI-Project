import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

# Config
batch_size = 32 # Größe eines Durchlaufs
img_height = 180 # Höhe des Bildes
img_width = 180  # Breite des Bildes

# Modeloutput
num_classes = 120 # Anzahl der Klassen bzw. Hunderassen

# Training
epochs = 100
learning_rate = 0.001


# Hochladen des Datensatzes
dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
data_dir = tf.keras.utils.get_file('Images', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Vorbereitung des Trainings- und Testdatensatzes
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  shuffle=True,
  seed=469,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Vorbereitung des Validierungsdatensatzes
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  shuffle=True,
  seed=469,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Namen der Klassen bzw. Hunderassen
class_names = train_ds.class_names
print(class_names)

# Anzeigen von 9 Hundebildern und deren Klassen
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Mischen und Zwischwenspeichern des Trainings- sowie Validierungsdatensatzes
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Überanpassung loswerden
data_augmentation = tf.keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


# Erstellen des Modells
input_tensor = layers.Input(shape=(img_width, img_height, 3))
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=input_tensor,
    input_shape=(img_width, img_height, 3),
    pooling='avg')

for layer in base_model.layers:
    layer.trainable = True  # trainable has to be false in order to freeze the layers
    
op = layers.Dense(256, activation='relu')(base_model.output)
op = layers.Dropout(.25)(op)

##
# softmax: calculates a probability for every possible class.
#
# activation='softmax': return the highest probability;
# for example, if 'Coat' is the highest probability then the result would be 
# something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
##
output_tensor = layers.Dense(num_classes, activation='softmax')(op)

model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)


# Training und Validierung
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
model.save('models/myModel')

# Visualisierung der Ergebnisse
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
