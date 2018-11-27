
from __future__ import print_function

import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import RMSprop, adam


# get encoded vectors and labels as pandas dataframe
feature_vectors_data = pd.read_csv("resources/encoded_vectors.csv", header=None)
feature_vectors_labels = pd.read_csv("resources/encoded_vectors_labels.csv", header=None)
print(feature_vectors_data.shape, feature_vectors_labels.shape)
print(type(feature_vectors_data), type(feature_vectors_labels))

# get encoded vectors and labels as numpy array (ndarray)
# DataFrame.values returns an ndarray
feature_vectors_data = feature_vectors_data.values
feature_vectors_labels = feature_vectors_labels.values
print(feature_vectors_data.shape, feature_vectors_labels.shape)
print(type(feature_vectors_data), type(feature_vectors_labels))

num_classes = 10

# the data, split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(feature_vectors_data, feature_vectors_labels, test_size=0.2)

print(y_train[0])

# standardizing: remove mean by scaling the unit variance

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Dense(512, input_shape=(128,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

ad = adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer=ad,
              metrics=['accuracy'])

from keras.callbacks import TensorBoard
## in terminal: tensorboard --logdir=/tmp/autoencoder
## then go to localhost:6006  in chrome
model.fit(x_train, y_train,
        epochs=30,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=[TensorBoard(log_dir='/tmp/mlp/run1-bigLRwithDropoutAll', histogram_freq=0, batch_size=128, write_graph=True,
                               write_grads=False, write_images=False, embeddings_freq=0,
                               embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                               update_freq=10000)])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])