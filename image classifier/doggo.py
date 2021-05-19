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
epochs = 30
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
  ]
)
#model von wem anders klauen
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width) + (3, ),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False



prediction_layer = layers.Dense(num_classes)
# Erstellen des Modells
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  base_model(training=False),
  layers.Dense(128, activation='relu'),
  prediction_layer
])


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
