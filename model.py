import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import glob
import shutil
import random
import os

physical_devices=tf.config.experimental.list_physical_devices('GPU')
print("Num of GPUs Available: ", len(physical_devices))
if len(physical_devices)>0:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  
import urllib.request

url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
filename = 'data/kagglecatsanddogs_5340.zip'

if not os.path.exists(filename):
    os.makedirs('data')
    urllib.request.urlretrieve(url, filename)
    print("Downloaded Successfully")

import zipfile

if not os.path.exists('data/PetImages'):
    if not os.path.exists('data/kagglecatsanddogs_5340.zip'):
        os.makedirs('data')
        url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
        urllib.request.urlretrieve(url, 'data/kagglecatsanddogs_5340.zip')
        print("data imported from kaggle")
    
    try:
        with zipfile.ZipFile('data/kagglecatsanddogs_5340.zip', 'r') as zip_ref:
            zip_ref.extractall('data')
    except zipfile.BadZipFile:
        print('Error: Invalid zip file')

os.chdir('data')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for c in random.sample(glob.glob('PetImages/Cat/*'), 500):
        shutil.move(c, 'train/cat')
    for c in random.sample(glob.glob('PetImages/Dog/*'), 500):
        shutil.move(c, 'train/dog')
    for c in random.sample(glob.glob('PetImages/Cat/*'), 100):
        shutil.move(c, 'valid/cat')
    for c in random.sample(glob.glob('PetImages/Dog/*'), 100):
        shutil.move(c, 'valid/dog')
    for c in random.sample(glob.glob('PetImages/Cat/*'), 50):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob('PetImages/Dog/*'), 50):
        shutil.move(c, 'test/dog')

os.chdir('../')

train_path='data/train/'
valid_path='data/train/'
test_path='data/test/'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

vgg16_model = tf.keras.applications.vgg16.VGG16()
print("vgg model loaded")

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable=False

model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
print("model compiled")
model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

model.save('model.h5')
os.rmdir("data/")