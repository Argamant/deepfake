from keras.applications import VGG16
#5.18
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np 
import matplotlib.pyplot as plt

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#physical_devices = tf.config.list_physical_devices('GPU') 
#try: 
#  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
#except: 
##  # Invalid device or cannot modify virtual devices once initialized. 
#  pass # dynamically grow GPU memory 


#feature extraction with a pretrained convolutional base.
conv_base = VGG16(weights = 'imagenet',
                    include_top = False,
                    input_shape= (150,150,3))
"""
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
"""

#base_dir = 'C:\\Users\\rugge\\source\\repos\\deepfake\\deepfake\\demo'
base_dir = '../demo'

train_dir = os.path.join(base_dir, 'train')
#what is the validation dir?
validation_dir = os.path.join(base_dir, 'test')
#test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4,4,512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size = batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i+1) * batch_size] = features_batch
        labels[i * batch_size : (i+1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

numTrainImages = 380
numTestImages = 140
train_features, train_labels = extract_features(train_dir, numTrainImages)
validation_features, validation_labels = extract_features(validation_dir, numTestImages)

train_features = np.reshape(train_features, (numTrainImages, 4 * 4 * 512))
print(validation_features)
validation_features = np.reshape(validation_features, (numTestImages, 4,4,512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim= 4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation= 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(lr=2e-5),
                loss='binary_crossentropy',
                metrics=['acc'])
   
history = model.fit(train_features, train_labels,
                    epochs = 5, 
                    batch_size = 20)
model.save('vgg16_model.h5')
#plotting the results

acc = history.history['acc']
#val_acc = history.history['val_acc']
loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs = range(1, len(acc) +1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
#plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()
