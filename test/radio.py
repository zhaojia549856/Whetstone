from __future__ import print_function

"""
Light-weight demo of SimpleSharpener, Spiking_BRelu, and Softmax_Decode for a fully connected net on mnist.
"""

import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adadelta
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.callbacks import SimpleSharpener, WhetstoneLogger
from neuro.model_printer import neuro
from keras import backend as K
import timeit

numClasses = 2
x_train = []
y_train = []

with open("training.txt", 'r') as f:
    for line in f.readlines():
        li = list(line.split(" "))
        x_train.append(li[1:-1])
        y_train.append(li[0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []

with open("testing.txt", 'r') as f:
    for line in f.readlines():
        li = list(line.split(" "))
        x_test.append(li[1:-1]) #-1 for new line
        y_test.append(li[0])

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(y_train, numClasses)
y_test = to_categorical(y_test, numClasses)


key = key_generator(num_classes=2, width=10, overlapping=False)


model = Sequential()
model.add(Dense(64, input_shape=(256,)))
model.add(Spiking_BRelu())
# model.add(Dense(512))
# model.add(Spiking_BRelu())
# model.add(Dense(64))
# model.add(Spiking_BRelu())
model.add(Dense(32))
model.add(Spiking_BRelu())
model.add(Dense(10))
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))

simple = SimpleSharpener(start_epoch=6, steps=5, epochs=True, bottom_up=True)

# Create a new directory to save the logs in.
log_dir = './simple_logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
time_delta = timeit.default_timer()
model.fit(x_train, y_train, batch_size=10, epochs=10, callbacks=[simple, logger])

time_delta = timeit.default_timer() - time_delta
num_neurons = 256 + 64 + 32 + 10 
num_synapses = 256 * 64 + 64 * 32 + 32 + 10

train_accuracy = model.evaluate(x_train, y_train)
test_accuracy = model.evaluate(x_test, y_test)


neuro(model, key, "danna2")
print(train_accuracy[1], test_accuracy[1], time_delta, num_neurons, num_synapses)
