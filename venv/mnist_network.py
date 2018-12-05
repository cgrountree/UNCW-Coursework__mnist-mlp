from __future__ import print_function

import time
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import RMSprop, adam

# get encoded vectors and labels as pandas dataframe
feature_vectors_data = pd.read_csv("resources/encoded_vectors.csv", header=None)
feature_vectors_labels = pd.read_csv("resources/encoded_vectors_labels.csv", header=None)

# get encoded vectors and labels as numpy array (ndarray)
# DataFrame.values returns an ndarray
feature_vectors_data = feature_vectors_data.values
feature_vectors_labels = feature_vectors_labels.values

# split train x and y data from the feature vector data and feature vector labels
x_train = feature_vectors_data[10000:50000]
y_train = feature_vectors_labels[10000:50000]

# split test x and y data from the feature vector data and feature vector labels
x_test = feature_vectors_data[:10000]
y_test = feature_vectors_labels[:10000]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print("Running...")

# set number of classes(labels for the dataset)
num_classes = 10

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

print(y_train.shape)
print(y_test.shape)

n_value = 2000
accuracy = []
time_in_seconds = []

#-----a for loop for testing inputs of different sizes in step 2000-----#

for i in range(2):
    model = Sequential()

    model.add(Dense(1024, input_shape=(128,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    ad = adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=ad,
                  metrics=['accuracy'])

    start_time = time.time()
    model.fit(x_train[:n_value], y_train[:n_value],
              verbose=1,
              epochs=100,
              batch_size=128,
              shuffle=True)

    time_in_seconds.append(time.time() - start_time)

    # score = model.evaluate(x_test, y_test, verbose=1)

    y_pred = model.predict(x_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy.append(accuracy_score(y_test, y_pred))
    n_value += 2000

accuracy = np.asarray(accuracy)
pd.DataFrame(accuracy).to_csv('accuracyvsnMNIST.csv', header=None, index=False)
time_in_seconds = np.asarray(time_in_seconds)
pd.DataFrame(time_in_seconds).to_csv('timevsnMNIST.csv', header=None,index=False)
print("Completed")

#-----below is code for a single run that is plotted within tensorboard-----#

# from keras.callbacks import TensorBoard
# ## in terminal: tensorboard --logdir=/tmp/mlp
# ## then go to localhost:6006  in chrome
# ## change log_dir to individual file names for each run.
# model.fit(x_train, y_train,
#         verbose=0,
#         epochs=100,
#         batch_size=128,
#         shuffle=True,
#         validation_data=(x_test, y_test),
#         callbacks=[TensorBoard(log_dir='/tmp/mlp/lr(.00001)_ep(100)', histogram_freq=0, batch_size=128, write_graph=True,
#                                write_grads=False, write_images=False, embeddings_freq=0,
#                                embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
#                                update_freq=10000)])
#
# score = model.evaluate(x_test, y_test, verbose=0)

# y_pred = model.predict(x_test)
# y_test = np.argmax(y_test, axis=1)
# y_pred = np.argmax(y_pred, axis=1)

# cm = confusion_matrix(y_test, y_pred)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print('Classifcation report:\n', classification_report(y_test, y_pred))
# print('Confusion matrix:\n', cm)