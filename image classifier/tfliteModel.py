import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

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


data = DataLoader.from_folder(data_dir)
train_data, validation_data = data.split(0.9)

model = image_classifier.create(train_data, validation_data=validation_data)

model.export(export_dir='./')